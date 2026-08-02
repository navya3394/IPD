import os
import sys
import time
from typing import List, Dict, Any

import cv2
import numpy as np
import torch
from ultralytics import YOLO


class PersonDetector:
    """Person detection using a YOLOv8 model.

    Args:
        model_variant: Model name (e.g., "yolov8n" for the nano version).
        conf_threshold: Minimum confidence for a detection to be kept.
        device: "cpu" or "cuda". If "auto", the class will pick CUDA when available.
    """

    def __init__(self, model_variant: str = "yolov8n", conf_threshold: float = 0.4, device: str = "auto"):
        self.model_variant = model_variant
        self.conf_threshold = float(conf_threshold)
        # Determine device
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        # Load model once – ultralytics will download the weights if missing.
        # The string ``f"{model_variant}.pt"`` works for built‑in models.
        try:
            self.model = YOLO(f"{self.model_variant}.pt")
        except Exception as e:
            raise RuntimeError(f"Failed to load YOLO model '{self.model_variant}': {e}")
        # Force the device for inference.
        self.model.to(self.device)
        self._last_latency_ms: float = 0.0

    def detect(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """Run person detection on a single frame.

        Returns a list of detections, each a dict with keys ``bbox`` (x1, y1, x2, y2) and ``confidence``.
        Only the "person" class (COCO class 0) is retained.
        """
        start = time.perf_counter()
        # YOLO expects BGR image; we already have that from OpenCV.
        results = self.model(frame, conf=self.conf_threshold, device=self.device, classes=[0])
        detections: List[Dict[str, Any]] = []
        for result in results:
            # result.boxes holds tensors for xyxy, conf, cls
            if result.boxes is None:
                continue
            boxes = result.boxes
            xyxy = boxes.xyxy.cpu().numpy()  # shape (N,4)
            conf = boxes.conf.cpu().numpy()
            cls = boxes.cls.cpu().numpy()
            for b, c, cl in zip(xyxy, conf, cls):
                if cl != 0:  # safety net; should already be filtered
                    continue
                if c < self.conf_threshold:
                    continue
                detections.append({
                    "bbox": [float(b[0]), float(b[1]), float(b[2]), float(b[3])],
                    "confidence": float(c),
                })
        self._last_latency_ms = (time.perf_counter() - start) * 1000.0
        return detections

    def get_last_latency_ms(self) -> float:
        """Return latency of the most recent ``detect`` call (ms)."""
        return float(self._last_latency_ms)


def _draw_boxes(frame: np.ndarray, detections: List[Dict[str, Any]]) -> np.ndarray:
    """Draw bounding boxes on the frame for visualisation.

    Modifies a copy of the input frame and returns it.
    """
    out = frame.copy()
    for det in detections:
        x1, y1, x2, y2 = map(int, det["bbox"])
        conf = det["confidence"]
        label = f"person: {conf:.2f}"
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(out, label, (x1, max(y1 - 5, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return out


def _process_image(detector: PersonDetector, img_path: str, out_dir: str) -> None:
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Unable to read image: {img_path}")
    detections = detector.detect(img)
    annotated = _draw_boxes(img, detections)
    basename = os.path.basename(img_path)
    out_path = os.path.join(out_dir, f"annotated_{basename}")
    cv2.imwrite(out_path, annotated)
    print(f"Processed image {basename}: {len(detections)} persons, latency {detector.get_last_latency_ms():.2f} ms")


def _process_video(detector: PersonDetector, vid_path: str, out_dir: str) -> None:
    cap = cv2.VideoCapture(vid_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Unable to open video: {vid_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    basename = os.path.splitext(os.path.basename(vid_path))[0]
    out_path = os.path.join(out_dir, f"{basename}_annotated.mp4")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
    total_frames = 0
    total_persons = 0
    latencies = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        detections = detector.detect(frame)
        latencies.append(detector.get_last_latency_ms())
        total_persons += len(detections)
        total_frames += 1
        annotated = _draw_boxes(frame, detections)
        writer.write(annotated)
    cap.release()
    writer.release()
    avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
    avg_persons = total_persons / total_frames if total_frames else 0.0
    print(f"Processed video {os.path.basename(vid_path)}: {total_frames} frames, avg persons/frame {avg_persons:.2f}, avg latency {avg_latency:.2f} ms")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python detector.py <image_or_video_path>")
        sys.exit(1)
    src_path = sys.argv[1]
    out_root = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(out_root, exist_ok=True)

    detector = PersonDetector()
    if os.path.isdir(src_path):
        # Process every image/video file in the directory.
        for entry in sorted(os.listdir(src_path)):
            full_path = os.path.join(src_path, entry)
            if os.path.isfile(full_path):
                ext = os.path.splitext(entry)[1].lower()
                if ext in {".jpg", ".jpeg", ".png", ".bmp"}:
                    _process_image(detector, full_path, out_root)
                elif ext in {".mp4", ".avi", ".mov", ".mkv"}:
                    _process_video(detector, full_path, out_root)
    else:
        ext = os.path.splitext(src_path)[1].lower()
        if ext in {".jpg", ".jpeg", ".png", ".bmp"}:
            _process_image(detector, src_path, out_root)
        elif ext in {".mp4", ".avi", ".mov", ".mkv"}:
            _process_video(detector, src_path, out_root)
        else:
            raise ValueError("Unsupported file type. Provide an image or video file.")
