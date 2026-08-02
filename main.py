# main.py – End‑to‑End pipeline (C1 → C5)

"""Pipeline driver that stitches together the first five components of the CCTV threat detection system:

1. **VideoIngestor** (C1) – reads and throttles frames.
2. **QualityGate** (C2) – assesses frame quality and optionally enhances.
3. **PersonDetector** (C3) – detects people and returns bounding boxes.
4. **PoseEstimator** (C4) – estimates 17‑point skeletons with a static‑frame reuse optimisation.
5. **PersonTracker** (C5) – assigns persistent track IDs across frames.

The script is deliberately defensive: each stage is wrapped in ``try/except`` so a single failure does not abort the whole video. Latencies for every stage are accumulated to report average timings at the end.

Annotated output is written to ``outputs/pipeline_test.mp4``.
"""

import os
import sys
import time
import random
from collections import defaultdict

import cv2
from typing import List, Dict, Tuple
import numpy as np

# Import pipeline components
from c1.ingest import VideoIngestor
from c2.quality_gate import QualityGate
from c3.detector import PersonDetector
from c4.pose_estimator import PoseEstimator
from c5.tracker import PersonTracker

# ---------------------------------------------------------------------
def _draw_skeleton(frame, keypoints):
    """Draw COCO‑style 17‑point skeleton on a BGR frame."""
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

# ---------------------------------------------------------------------

def _draw_tracks(frame: np.ndarray, detections: List[Dict], id_colors: Dict[int, Tuple[int, int, int]]) -> np.ndarray:
    """Draw bounding boxes with track IDs on the frame.

    ``id_colors`` maps a ``track_id`` to a BGR colour tuple. New IDs are
    assigned a random colour the first time they appear.
    """
    for det in detections:
        bbox = det["bbox"]
        tid = det.get("track_id")
        if tid is None:
            # Skip detections without a track_id; this should not happen but guards against errors.
            continue
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
# ---------------------------------------------------------------------
def main():
    if len(sys.argv) < 2:
        print("Usage: python main.py <video_path> [target_fps]")
        sys.exit(1)
    video_path = sys.argv[1]
    target_fps = int(sys.argv[2]) if len(sys.argv) > 2 else 15

    # Initialise components
    ingestor = VideoIngestor(video_path, target_fps)
    quality_gate = QualityGate()  # defaults to brisque method
    detector = PersonDetector()
    pose_estimator = PoseEstimator()
    tracker = PersonTracker()

    # Output video setup
    os.makedirs("outputs", exist_ok=True)
    out_path = os.path.join("outputs", "pipeline_test.mp4")
    writer = None  # will be created after first frame for proper size

    # Statistics containers
    stats = defaultdict(float)  # cumulative latency per stage (ms)
    frame_counts = defaultdict(int)
    total_frames = 0
    unique_track_ids = set()
    prev_bboxes = []  # for PoseEstimator reuse optimisation
    id_colors = {}

    for frame_dict in ingestor.get_frame_stream():
        total_frames += 1
        frame_id = frame_dict["frame_id"]
        frame = frame_dict["frame"]

        # Initialise writer on first frame
        if writer is None:
            h, w = frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(out_path, fourcc, target_fps, (w, h))

        # ---------- Stage 1 – Quality Gate ----------
        try:
            start = time.time()
            qg_out = quality_gate.process(frame)
            latency = (time.time() - start) * 1000.0
            stats["quality_gate"] += latency
            frame_counts["quality_gate"] += 1
            frame = qg_out["frame"]
        except Exception as e:
            print(f"[Error][frame {frame_id}] QualityGate failed: {e}")

        # ---------- Stage 2 – Person Detection ----------
        detections = []
        try:
            start = time.time()
            detections = detector.detect(frame)
            latency = (time.time() - start) * 1000.0
            stats["detection"] += latency
            frame_counts["detection"] += 1
        except Exception as e:
            print(f"[Error][frame {frame_id}] PersonDetector failed: {e}")

        # ---------- Stage 3 – Pose Estimation ----------
        pose_results = []
        try:
            start = time.time()
            bboxes = [{"bbox": d["bbox"]} for d in detections]
            pose_results = pose_estimator.estimate(frame, bboxes, previous_bboxes=prev_bboxes)
            latency = (time.time() - start) * 1000.0
            stats["pose"] += latency
            frame_counts["pose"] += 1
            prev_bboxes = pose_results
        except Exception as e:
            print(f"[Error][frame {frame_id}] PoseEstimator failed: {e}")

        # ---------- Stage 4 – Tracking ----------
        tracked = []
        try:
            start = time.time()
            tracked = tracker.update(frame_id, detections)
            latency = (time.time() - start) * 1000.0
            stats["tracking"] += latency
            frame_counts["tracking"] += 1
            # Merge track IDs into pose results for drawing
            for pr in pose_results:
                for det in tracked:
                    if det["bbox"] == pr["bbox"]:
                        pr["track_id"] = det["track_id"]
                        unique_track_ids.add(det["track_id"])
                        break
        except Exception as e:
            print(f"[Error][frame {frame_id}] PersonTracker failed: {e}")

        # ---------- Drawing ----------
        if detections:
            frame = _draw_tracks(frame, tracked, id_colors)
        for pr in pose_results:
            frame = _draw_skeleton(frame, pr["keypoints"])
            if "track_id" in pr:
                x1, y1, x2, y2 = pr["bbox"]
                cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                cv2.putText(frame, f"ID {pr['track_id']}", (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        writer.write(frame)

    # Release resources
    if writer is not None:
        writer.release()

    # Summary reporting
    print("--- Pipeline Summary ---")
    print(f"Total frames processed: {total_frames}")
    for stage in ["quality_gate", "detection", "pose", "tracking"]:
        cnt = frame_counts.get(stage, 0)
        if cnt:
            avg = stats[stage] / cnt
            print(f"Average {stage} latency: {avg:.2f} ms (over {cnt} frames)")
    total_avg = sum(stats[s] for s in ["quality_gate", "detection", "pose", "tracking"]) / max(total_frames, 1)
    print(f"Average end‑to‑end latency per frame: {total_avg:.2f} ms")
    print(f"Total unique track IDs seen: {len(unique_track_ids)}")

if __name__ == "__main__":
    main()
