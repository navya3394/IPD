# c4/pose_estimator.py
"""Pose Estimation module (C4)

Implements a thin wrapper around the Ultralytics YOLOv8 pose model.
The class is deliberately lightweight – it expects a list of person
bounding boxes (the output of the C3 PersonDetector) and returns a list
of dictionaries that contain the original bbox, 17 key‑point coordinates
and optional reuse information.

Key features
------------
* **Model variant** – defaults to ``yolov8n-pose`` which is the smallest
  HW‑tier model.
* **Device auto‑detect** – ``cuda`` when ``torch.cuda.is_available()``
  otherwise ``cpu``.
* **Skip‑if‑static optimisation** – if a bbox centre moves less than
  ``delta_threshold`` pixels compared with the previous frame, the
  previously computed key‑points are reused and the entry is marked with
  ``"reused": True``.
* **Latency logging** – ``get_last_latency_ms()`` returns the processing
  time of the most recent ``estimate`` call.
* **Command‑line demo** – runs a C3‑style person detector on an image or
  video, performs pose estimation, draws a skeleton and writes the
  annotated output to ``c4/output/``.
"""

from __future__ import annotations

import os
import sys
import time
from typing import List, Optional, Dict

import cv2
import numpy as np
import torch
from ultralytics import YOLO

# Import the C3 detector – relative import works because both c3 and c4 are
# top‑level packages in the repository.
from c3.detector import PersonDetector

class PoseEstimator:
    """Wraps a YOLOv8 pose model.

    Parameters
    ----------
    model_variant: str, optional
        YOLOv8 pose model name – e.g. ``"yolov8n-pose"`` (default).
    device: str, optional
        ``"cpu"`` or ``"cuda"``. If omitted the best available device is
        selected automatically.
    """

    def __init__(self, model_variant: str = "yolov8n-pose", device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        # Load the model once – the Ultralytics API loads weights on first use.
        self.model = YOLO(model_variant).to(self.device)
        self._last_latency_ms: float = 0.0

    # ---------------------------------------------------------------------
    def estimate(
        self,
        frame: np.ndarray,
        bboxes: List[Dict],
        previous_bboxes: Optional[List[Dict]] = None,
        delta_threshold: float = 5.0,
    ) -> List[Dict]:
        """Estimate pose for each detection.

        Parameters
        ----------
        frame: np.ndarray
            BGR image as returned by OpenCV.
        bbbox: list[dict]
            Each dict must contain a ``"bbox"`` key with ``[x1, y1, x2, y2]``.
        previous_bboxes: list[dict] | None
            Same structure as ``bboxes`` from the previous call. Only the
            ``"bbox"`` and ``"keypoints"`` entries are consulted for reuse.
        delta_threshold: float
            Maximum centre‑point displacement (in pixels) that allows reuse.

        Returns
        -------
        list[dict]
            Each entry contains ``bbox``, ``keypoints`` (17×[x, y, conf]),
            ``track_hint_id`` (always ``None`` here) and optionally ``reused``.
        """
        start = time.time()

        results: List[Dict] = []
        # Helper to compute centre of a bbox.
        def centre(b: List[float]) -> np.ndarray:
            x1, y1, x2, y2 = b
            return np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0])

        for idx, cur in enumerate(bboxes):
            cur_bbox = cur["bbox"]
            reuse = False
            # Reuse logic ---------------------------------------------------
            if (
                previous_bboxes
                and idx < len(previous_bboxes)
                and "keypoints" in previous_bboxes[idx]
            ):
                prev_bbox = previous_bboxes[idx]["bbox"]
                if np.linalg.norm(centre(cur_bbox) - centre(prev_bbox)) < delta_threshold:
                    # Reuse previous keypoints.
                    keypoints = previous_bboxes[idx]["keypoints"]
                    reuse = True
            # Compute fresh keypoints if not reused -----------------------
            if not reuse:
                # Crop ROI to reduce compute – the pose model can operate on
                # the whole frame, but cropping improves speed for small
                # detections.
                x1, y1, x2, y2 = map(int, cur_bbox)
                roi = frame[y1:y2, x1:x2]
                if roi.size > 0:
                    # Ultralytics expects RGB images.
                    roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
                    # Predict – we keep default confidence thresholds.
                    pose_res = self.model.predict(source=roi_rgb, verbose=False)[0]
                    # The model returns keypoints relative to the ROI.
                    # Safely handle keypoints extraction across Ultralytics versions
                    if pose_res.keypoints is not None and getattr(pose_res.keypoints, 'has_visible', True):
                        kp_array = None
                        # Prefer .xy/.conf if available (newer versions)
                        if hasattr(pose_res.keypoints, 'xy') and hasattr(pose_res.keypoints, 'conf'):
                            try:
                                xy = pose_res.keypoints.xy[0].cpu().numpy()
                                conf = pose_res.keypoints.conf[0].cpu().numpy().reshape(-1, 1)
                                kp_array = np.hstack([xy, conf])
                            except Exception:
                                kp_array = None
                        # Fallback to .data tensor
                        if kp_array is None and hasattr(pose_res.keypoints, 'data'):
                            try:
                                kp_array = pose_res.keypoints.data[0].cpu().numpy()
                            except Exception:
                                kp_array = None
                        # Final fallback: convert the object directly to numpy if possible
                        if kp_array is None:
                            try:
                                kp_array = np.array(pose_res.keypoints)
                            except Exception:
                                kp_array = None
                        if kp_array is None:
                            kp_array = np.zeros((17, 3))
                        # Translate coordinates back to the original frame.
                        kp_array[:, 0] += x1
                        kp_array[:, 1] += y1
                        keypoints = kp_array.tolist()
                    else:
                        keypoints = [[0.0, 0.0, 0.0] for _ in range(17)]
                else:
                    # No pose detected – fill with zeros.
                    keypoints = [[0.0, 0.0, 0.0] for _ in range(17)]
            # Append result for this detection.
            entry: Dict = {
                "bbox": cur_bbox,
                "keypoints": keypoints,
                "track_hint_id": None,
            }
            if reuse:
                entry["reused"] = True
            results.append(entry)

        self._last_latency_ms = (time.time() - start) * 1000.0
        return results

    # ---------------------------------------------------------------------
    def get_last_latency_ms(self) -> float:
        """Return latency of the most recent ``estimate`` call (ms)."""
        return self._last_latency_ms


# -------------------------------------------------------------------------
def _draw_skeleton(frame: np.ndarray, keypoints: List[List[float]]) -> np.ndarray:
    """Draw 17‑point skeleton on a BGR frame.

    The COCO skeleton connection list is used – it is a static list of
    (start, end) index pairs (0‑based).
    """
    # COCO pose connections (17 points).
    skeleton = [
        (0, 1), (1, 2), (2, 3), (3, 4),      # right arm
        (0, 5), (5, 6), (6, 7), (7, 8),      # left arm
        (0, 9), (9, 10), (10, 11), (11, 12),  # right leg
        (0, 13), (13, 14), (14, 15), (15, 16),  # left leg
    ]
    for (x, y, conf) in keypoints:
        if conf > 0:
            cv2.circle(frame, (int(x), int(y)), 3, (0, 255, 0), -1)
    for i, j in skeleton:
        if keypoints[i][2] > 0 and keypoints[j][2] > 0:
            pt1 = (int(keypoints[i][0]), int(keypoints[i][1]))
            pt2 = (int(keypoints[j][0]), int(keypoints[j][1]))
            cv2.line(frame, pt1, pt2, (255, 0, 0), 2)
    return frame


def _process_video(path: str, detector: PersonDetector, estimator: PoseEstimator) -> None:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Unable to open video file: {path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    basename = os.path.splitext(os.path.basename(path))[0]
    out_path = os.path.join("c4", "output", f"{basename}_pose.mp4")
    out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    total_latency = 0.0
    reuse_count = 0
    frame_cnt = 0
    prev_boxes: Optional[List[Dict]] = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_cnt += 1
        # Person detection (C3)
        detections = detector.detect(frame)
        bboxes = [{"bbox": d["bbox"]} for d in detections]
        pose_out = estimator.estimate(frame, bboxes, previous_bboxes=prev_boxes)
        # Update reuse statistics
        for p in pose_out:
            if p.get("reused"):
                reuse_count += 1
        # Draw skeletons
        for p in pose_out:
            frame = _draw_skeleton(frame, p["keypoints"])
        out.write(frame)
        total_latency += estimator.get_last_latency_ms()
        # Keep current boxes for next frame reuse check
        prev_boxes = pose_out

    cap.release()
    out.release()
    avg_latency = total_latency / max(frame_cnt, 1)
    reuse_percent = (reuse_count / max(frame_cnt, 1)) * 100.0
    print(f"[C4] Processed {frame_cnt} frames – avg latency {avg_latency:.2f} ms, reuse {reuse_percent:.1f}%")
    print(f"Annotated video saved to: {out_path}")


def _process_image(path: str, detector: PersonDetector, estimator: PoseEstimator) -> None:
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Unable to read image file: {path}")
    detections = detector.detect(img)
    bboxes = [{"bbox": d["bbox"]} for d in detections]
    pose_out = estimator.estimate(img, bboxes)
    for p in pose_out:
        img = _draw_skeleton(img, p["keypoints"])
    basename = os.path.splitext(os.path.basename(path))[0]
    out_path = os.path.join("c4", "output", f"{basename}_pose.jpg")
    cv2.imwrite(out_path, img)
    print(f"[C4] Annotated image saved to: {out_path}")


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python c4/pose_estimator.py <image_or_video_path>")
        sys.exit(1)
    src_path = sys.argv[1]
    if PersonDetector is None:
        raise RuntimeError("PersonDetector from C3 could not be imported – ensure C3 is built first.")
    detector = PersonDetector()
    estimator = PoseEstimator()
    ext = os.path.splitext(src_path)[1].lower()
    if ext in {".mp4", ".avi", ".mov", ".mkv"}:
        _process_video(src_path, detector, estimator)
    else:
        _process_image(src_path, detector, estimator)


if __name__ == "__main__":
    main()
