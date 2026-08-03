# c5/tracker.py
"""Person tracking (C5) module.

This implementation wraps a lightweight **IoU‑based association** algorithm.
We considered two options:

1. **Ultralytics built‑in tracking** (`model.track(..., tracker="bytetrack.yaml")`).
   This requires running the YOLO detector again inside the tracker, which
   would duplicate the work already performed by the C3 `PersonDetector`.
   It also returns a different result structure, making it harder to keep
   a consistent API with the rest of the pipeline.

2. **Custom in‑memory tracker** that accepts the C3 detection output and
   assigns persistent `track_id`s using IoU matching and a simple
   "lost‑frame" policy.

We chose **option 2** because it integrates cleanly with the C3 detection
format, keeps the processing lightweight, and satisfies the requirement of
maintaining an in‑memory state without external services.
"""

from __future__ import annotations

import os
import random
import sys
import time
from collections import defaultdict
from typing import List, Dict, Tuple

try:
    import cv2
    _has_cv2 = True
except ImportError:
    print("[WARNING] opencv-python (cv2) not installed – tracker demo will not run.")
    _has_cv2 = False
import numpy as np

# Import the C3 detector for the demo CLI.
try:
    from c3.detector import PersonDetector
except Exception as e:  # pragma: no cover
    PersonDetector = None
    print("[WARNING] Could not import C3 detector – demo will fail if used.", e)


def _iou(box_a: List[float], box_b: List[float]) -> float:
    """Calculate Intersection‑over‑Union for two boxes.

    Boxes are in ``[x1, y1, x2, y2]`` format.
    """
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter = inter_w * inter_h
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = max(area_a + area_b - inter, 1e-6)
    return inter / union


class PersonTracker:
    """Simple IoU‑based multi‑object tracker.

    Parameters
    ----------
    max_lost_frames : int, optional
        Number of consecutive frames a track may be missing before it is
        removed (default ``30``).
    """

    def __init__(self, max_lost_frames: int = 30):
        self.max_lost = max_lost_frames
        self.next_id: int = 1
        # Track state: id -> dict with keys ``bbox``, ``last_seen``, ``lost``
        self.tracks: Dict[int, Dict] = {}
        self.last_latency_ms: float = 0.0

    # ------------------------------------------------------------------
    def update(self, frame_id: int, detections: List[Dict]) -> List[Dict]:
        """Update tracker with detections from the current frame.

        Each detection dict must contain a ``"bbox"`` entry formatted as
        ``[x1, y1, x2, y2]``. The method returns a new list where each dict
        is enriched with a persistent ``track_id``.
        """
        start = time.time()
        # Step 1 – compute IoU matrix between existing tracks and new detections.
        track_ids = list(self.tracks.keys())
        iou_matrix: List[List[float]] = []
        for tid in track_ids:
            iou_row = []
            t_bbox = self.tracks[tid]["bbox"]
            for det in detections:
                iou_row.append(_iou(t_bbox, det["bbox"]))
            iou_matrix.append(iou_row)

        # Step 2 – greedy matching (highest IoU first).
        matched_tracks: Dict[int, int] = {}  # track_id -> detection index
        matched_dets: set[int] = set()
        if iou_matrix:
            # Flatten and sort by IoU descending.
            flat = [
                (i, j, iou_matrix[i][j])
                for i in range(len(track_ids))
                for j in range(len(detections))
                if iou_matrix[i][j] > 0.3
            ]
            flat.sort(key=lambda x: x[2], reverse=True)
            for ti, di, _ in flat:
                if ti in matched_tracks or di in matched_dets:
                    continue
                matched_tracks[track_ids[ti]] = di
                matched_dets.add(di)

        # Step 3 – update matched tracks.
        for tid, det_idx in matched_tracks.items():
            det = detections[det_idx]
            self.tracks[tid]["bbox"] = det["bbox"]
            self.tracks[tid]["last_seen"] = frame_id
            self.tracks[tid]["lost"] = 0
            det["track_id"] = tid

        # Step 4 – handle unmatched detections → create new tracks.
        for idx, det in enumerate(detections):
            if idx in matched_dets:
                continue
            tid = self.next_id
            self.next_id += 1
            self.tracks[tid] = {
                "bbox": det["bbox"],
                "last_seen": frame_id,
                "lost": 0,
            }
            det["track_id"] = tid

        # Step 5 – increase lost counter for tracks not seen in this frame.
        for tid in list(self.tracks.keys()):
            if self.tracks[tid]["last_seen"] != frame_id:
                self.tracks[tid]["lost"] += 1
                if self.tracks[tid]["lost"] > self.max_lost:
                    del self.tracks[tid]

        self.last_latency_ms = (time.time() - start) * 1000.0
        return detections

    def get_last_latency_ms(self) -> float:
        """Return the latency (in ms) of the most recent ``update`` call."""
        return self.last_latency_ms

    # ------------------------------------------------------------------
    def get_active_track_ids(self) -> List[int]:
        """Return a list of currently active track IDs."""
        return list(self.tracks.keys())


# -------------------------------------------------------------------------
def _draw_tracks(frame: np.ndarray, detections: List[Dict], id_colors: Dict[int, Tuple[int, int, int]]) -> np.ndarray:
    """Draw bounding boxes with track IDs on the frame.

    ``id_colors`` maps a ``track_id`` to a BGR colour tuple. New IDs are
    assigned a random colour the first time they appear.
    """
    for det in detections:
        bbox = det["bbox"]
        tid = det["track_id"]
        if tid not in id_colors:
            id_colors[tid] = tuple(random.randint(0, 255) for _ in range(3))
        color = id_colors[tid]
        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 4)
        cv2.putText(
            frame,
            f"ID {tid}",
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2,
        )
    return frame


def _run_demo(video_path: str) -> None:
    if PersonDetector is None:
        print("[WARNING] C3 PersonDetector not available – using dummy detector (no detections).")
        class DummyDetector:
            def detect(self, frame):
                return []  # no detections
        detector = DummyDetector()
    else:
        detector = PersonDetector()
    tracker = PersonTracker()
    if not _has_cv2:
        print("[WARNING] opencv-python (cv2) not available – cannot run video processing demo.")
        return
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Unable to open video file: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    base = os.path.splitext(os.path.basename(video_path))[0]
    out_path = os.path.join("c5", "output", f"{base}_tracked.mp4")
    out = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

    frame_id = 0
    total_tracks = set()
    tracks_per_frame = []
    id_colors: Dict[int, Tuple[int, int, int]] = {}

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_id += 1
        detections = detector.detect(frame)
        # C3 detection format: list of dicts with "bbox" and "confidence".
        tracked = tracker.update(frame_id, detections)
        total_tracks.update([d["track_id"] for d in tracked])
        tracks_per_frame.append(len(tracked))
        frame = _draw_tracks(frame, tracked, id_colors)
        out.write(frame)

    cap.release()
    out.release()
    avg_tracks = sum(tracks_per_frame) / max(len(tracks_per_frame), 1)
    print(f"[C5] Finished processing {frame_id} frames.")
    print(f"Unique track IDs assigned: {len(total_tracks)}")
    print(f"Average tracks per frame: {avg_tracks:.2f}")
    print(f"Annotated video saved to: {out_path}")


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python c5/tracker.py <video_path>")
        sys.exit(1)
    _run_demo(sys.argv[1])


if __name__ == "__main__":
    main()
