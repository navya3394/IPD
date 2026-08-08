# main_demo.py
"""Offline end-to-end C1-C8 demo pipeline."""

import argparse
import os
import random
import sys
import time
from collections import defaultdict
from typing import Dict, List, Tuple

import cv2
import numpy as np

from c1.ingest import VideoIngestor
from c2.quality_gate import QualityGate
from c3.detector import PersonDetector
from c4.pose_estimator import PoseEstimator
from c5.tracker import PersonTracker
from c6.binary_action_classifier import BinaryActionClassifier
from c7.temporal_engine import TemporalEngine
from c8.trust_engine import TrustScoreEngine
from c8.crowd_density import CrowdDensityEstimator


def _draw_skeleton(frame, keypoints):
    """Draw the same COCO-style 17-point skeleton used by main.py."""
    skeleton = [
        (0, 1), (1, 2), (2, 3), (3, 4),
        (0, 5), (5, 6), (6, 7), (7, 8),
        (0, 9), (9, 10), (10, 11), (11, 12),
        (0, 13), (13, 14), (14, 15), (15, 16),
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


def _draw_tracks(
    frame: np.ndarray,
    detections: List[Dict],
    id_colors: Dict[int, Tuple[int, int, int]],
) -> np.ndarray:
    """Draw the same bounding boxes and track IDs used by main.py."""
    for det in detections:
        bbox = det["bbox"]
        tid = det.get("track_id")

        if tid is None:
            continue

        if tid not in id_colors:
            id_colors[tid] = tuple(
                random.randint(0, 255) for _ in range(3)
            )

        color = id_colors[tid]
        x1, y1, x2, y2 = map(int, bbox)

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            color,
            4,
        )

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


def _get_raw_frame_count(video_path: str) -> int:
    """Read raw frame count without decoding the video."""
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise RuntimeError(
            f"Failed to open video with cv2: {video_path}"
        )

    try:
        return int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        cap.release()


def map_c6_label_to_action(label: str) -> str:
    """Map the C6 binary label to a TemporalEngine action token."""
    return "punch" if label.lower() == "fighting" else "normal"


def _run_pipeline_pass(
    video_path: str,
    target_fps: int,
    qg: QualityGate,
    detector: PersonDetector,
    estimator: PoseEstimator,
    tracker: PersonTracker,
):
    """Run C1-C5 once, collecting tracking/drawing data and latency stats."""

    vi = VideoIngestor(
        video_path,
        target_fps=target_fps,
    )

    stats = defaultdict(float)
    frame_counts = defaultdict(int)

    unique_track_ids = set()
    track_appearances = defaultdict(int)

    frames_for_output = []
    prev_pose_bboxes: List[Dict] = []

    for frame_dict in vi.get_frame_stream():
        frame_id = frame_dict["frame_id"]
        timestamp = frame_dict["timestamp"]
        frame = frame_dict["frame"]

        # ---------- Stage 1 - Quality Gate ----------
        try:
            start = time.perf_counter()

            qg_out = qg.process(frame)

            latency = (time.perf_counter() - start) * 1000.0
            stats["quality_gate"] += latency
            frame_counts["quality_gate"] += 1

            frame = qg_out["frame"]

        except Exception as e:
            print(
                f"[Error][frame {frame_id}] "
                f"QualityGate failed: {e}"
            )

        # ---------- Stage 2 - Person Detection ----------
        detections = []

        try:
            start = time.perf_counter()

            detections = detector.detect(frame)

            latency = (time.perf_counter() - start) * 1000.0
            stats["detection"] += latency
            frame_counts["detection"] += 1

        except Exception as e:
            print(
                f"[Error][frame {frame_id}] "
                f"PersonDetector failed: {e}"
            )

        # ---------- Stage 3 - Pose Estimation ----------
        pose_results = []

        try:
            start = time.perf_counter()

            bboxes = [
                {"bbox": d["bbox"]}
                for d in detections
            ]

            pose_results = estimator.estimate(
                frame,
                bboxes,
                previous_bboxes=prev_pose_bboxes,
            )

            latency = (time.perf_counter() - start) * 1000.0
            stats["pose"] += latency
            frame_counts["pose"] += 1

            prev_pose_bboxes = pose_results

        except Exception as e:
            print(
                f"[Error][frame {frame_id}] "
                f"PoseEstimator failed: {e}"
            )

        # ---------- Stage 4 - Tracking ----------
        tracked = []

        try:
            start = time.perf_counter()

            tracked = tracker.update(
                frame_id,
                detections,
            )

            latency = (time.perf_counter() - start) * 1000.0
            stats["tracking"] += latency
            frame_counts["tracking"] += 1

            for det in tracked:
                tid = det["track_id"]

                unique_track_ids.add(tid)
                track_appearances[tid] += 1

            # Merge track IDs into pose results.
            for pr in pose_results:
                for det in tracked:
                    if det["bbox"] == pr["bbox"]:
                        pr["track_id"] = det["track_id"]
                        break

        except Exception as e:
            print(
                f"[Error][frame {frame_id}] "
                f"PersonTracker failed: {e}"
            )

        frames_for_output.append(
            {
                "frame_id": frame_id,
                "timestamp": timestamp,
                "frame": frame.copy(),
                "tracked": tracked,
                "pose_results": pose_results,
            }
        )

    return (
        frames_for_output,
        stats,
        frame_counts,
        unique_track_ids,
        track_appearances,
    )


def _write_output_video(
    frames_for_output,
    out_path: str,
    target_fps: int,
    trust_scores: Dict[int, float],
):
    """Write annotated frames using main.py's VideoWriter pattern."""

    writer = None
    id_colors = {}

    try:
        for item in frames_for_output:
            frame = item["frame"].copy()
            tracked = item["tracked"]
            pose_results = item["pose_results"]

            # Same writer initialization pattern as main.py.
            if writer is None:
                h, w = frame.shape[:2]

                fourcc = cv2.VideoWriter_fourcc(*"mp4v")

                writer = cv2.VideoWriter(
                    out_path,
                    fourcc,
                    target_fps,
                    (w, h),
                )

                if not writer.isOpened():
                    raise RuntimeError(
                        f"Failed to open VideoWriter: {out_path}"
                    )

            # Bounding boxes + IDs.
            if tracked:
                frame = _draw_tracks(
                    frame,
                    tracked,
                    id_colors,
                )

            # Skeletons.
            for pr in pose_results:
                frame = _draw_skeleton(
                    frame,
                    pr["keypoints"],
                )

            # Static final trust score.
            # C8 is intentionally evaluated only once after tracking.
            for det in tracked:
                tid = det["track_id"]

                x1, y1, x2, y2 = map(
                    int,
                    det["bbox"],
                )

                cv2.putText(
                    frame,
                    f"Trust: {trust_scores.get(tid, 1.0):.2f}",
                    (x1, max(20, y2 + 25)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2,
                )

            writer.write(frame)

    finally:
        if writer is not None:
            writer.release()


def main(
    video_path: str,
    target_fps: int = 15,
) -> None:

    os.makedirs("outputs", exist_ok=True)

    out_path = os.path.join(
        "outputs",
        "demo_pipeline_output.mp4",
    )

    # ------------------------------------------------------------------
    # Raw frame count.
    #
    # This intentionally does NOT iterate through VideoIngestor.
    # VideoIngestor may throttle frames according to target_fps.
    # ------------------------------------------------------------------
    raw_total_frames = _get_raw_frame_count(
        video_path
    )

    if raw_total_frames <= 0:
        raise RuntimeError(
            f"Could not determine a valid raw frame count for: "
            f"{video_path}"
        )

    start_clip_frame = int(
        0.25 * raw_total_frames
    )

    midpoint_frame = int(
        raw_total_frames * 0.5
    )

    # ------------------------------------------------------------------
    # C6 - Run exactly once.
    # ------------------------------------------------------------------
    classifier = BinaryActionClassifier()

    c6_result = classifier.classify_clip(
        video_path,
        start_frame=start_clip_frame,
    )

    action_label = map_c6_label_to_action(
        c6_result["label"]
    )

    # ------------------------------------------------------------------
    # First pass: C1-C5.
    #
    # This pass determines the primary track and stores the data needed
    # for the final offline output video.
    # ------------------------------------------------------------------
    qg = QualityGate()
    detector = PersonDetector()
    estimator = PoseEstimator()
    tracker = PersonTracker()

    (
        frames_for_output,
        stats,
        frame_counts,
        unique_track_ids,
        track_appearances,
    ) = _run_pipeline_pass(
        video_path,
        target_fps,
        qg,
        detector,
        estimator,
        tracker,
    )

    total_frames_processed = len(
        frames_for_output
    )

    if not track_appearances:
        raise RuntimeError(
            "No track IDs were observed; "
            "cannot construct the C6-C7-C8 event."
        )

    # Most frequently visible track = primary track.
    primary_track_id = max(
        track_appearances,
        key=lambda tid: (
            track_appearances[tid],
            -tid,
        ),
    )

    # ------------------------------------------------------------------
    # Find the ingested frame nearest to the raw midpoint.
    # ------------------------------------------------------------------
    midpoint_item = min(
        frames_for_output,
        key=lambda item: abs(
            item["frame_id"] - midpoint_frame
        ),
    )

    midpoint_timestamp = float(
        midpoint_item["timestamp"]
    )

    midpoint_context_frame = midpoint_item[
        "frame"
    ]

    # ------------------------------------------------------------------
    # C8 Crowd Density.
    #
    # Exactly one call, using the midpoint frame.
    # ------------------------------------------------------------------
    crowd = CrowdDensityEstimator()

    crowd_start = time.perf_counter()

    crowd_ctx = crowd.estimate(
        midpoint_context_frame
    )

    crowd_latency_ms = (
        time.perf_counter() - crowd_start
    ) * 1000.0

    # ------------------------------------------------------------------
    # C6 -> C7 -> C8.
    #
    # This is deliberately NOT wrapped in try/except.
    # ------------------------------------------------------------------
    temporal = TemporalEngine()
    trust = TrustScoreEngine()

    action_event = {
        "track_id": primary_track_id,
        "action_label": action_label,
        "confidence": c6_result["confidence"],
        "frame_id": midpoint_frame,
        "timestamp": midpoint_timestamp,
    }

    # C7 - exactly once.
    temporal_result = temporal.update(
        action_event
    )

    temporal_latency_ms = (
        temporal.get_last_latency_ms()
    )

    # C8 - exactly once.
    trust_context = dict(crowd_ctx)
    trust_context["zone"] = "public"

    trust_result = trust.update(
        primary_track_id,
        temporal_result,
        trust_context,
    )

    trust_latency_ms = (
        trust.get_last_latency_ms()
    )

    # ------------------------------------------------------------------
    # Final static trust values for the offline output.
    #
    # Every track starts at 1.0. Only the primary track receives the
    # single final C8 result.
    # ------------------------------------------------------------------
    trust_scores = {
        tid: 1.0
        for tid in unique_track_ids
    }

    trust_scores[primary_track_id] = (
        trust_result["trust_score"]
    )

    # ------------------------------------------------------------------
    # Write annotated video.
    # ------------------------------------------------------------------
    _write_output_video(
        frames_for_output,
        out_path,
        target_fps,
        trust_scores,
    )

    # ------------------------------------------------------------------
    # Final terminal summary.
    # ------------------------------------------------------------------
    video_name = os.path.basename(
        video_path
    )

    print("\n=== Offline C1-C8 Demo Summary ===")

    print(
        f"Video                  : {video_name}"
    )

    print(
        f"Total frames processed : "
        f"{total_frames_processed}"
    )

    print(
        f"Raw video frame count  : "
        f"{raw_total_frames}"
    )

    print(
        f"Unique track IDs seen  : "
        f"{len(unique_track_ids)}"
    )

    print(
        f"Primary track ID       : "
        f"{primary_track_id}"
    )

    print("\n--- C6 Binary Action ---")

    print(
        f"Raw label              : "
        f"{c6_result['label']}"
    )

    print(
        f"Confidence             : "
        f"{c6_result['confidence']:.4f}"
    )

    print(
        f"Mapped action          : "
        f"{action_label}"
    )

    print("\n--- C7 Temporal Engine ---")

    print(
        f"Suspicion score        : "
        f"{temporal_result['suspicion_score']:.4f}"
    )

    print(
        f"Matched pattern        : "
        f"{temporal_result['matched_pattern'] or 'None'}"
    )

    print("\n--- C8 Trust Engine ---")

    print(
        f"Trust score            : "
        f"{trust_result['trust_score']:.4f}"
    )

    print(
        f"Alert                  : "
        f"{trust_result['alert']}"
    )

    print(
        f"Threshold used         : "
        f"{trust_result['threshold_used']:.4f}"
    )

    print("\n--- Per-stage latency ---")

    for stage in [
        "quality_gate",
        "detection",
        "pose",
        "tracking",
    ]:
        count = frame_counts.get(
            stage,
            0,
        )

        if count:
            avg = (
                stats[stage] / count
            )

            print(
                f"{stage:20s}: "
                f"{avg:.2f} ms average "
                f"({count} frames)"
            )
        else:
            print(
                f"{stage:20s}: "
                f"N/A (0 successful frames)"
            )

    print(
        f"{'C6 classify_clip':20s}: "
        f"{c6_result['latency_ms']:.2f} ms"
    )

    print(
        f"{'C7 temporal.update':20s}: "
        f"{temporal_latency_ms:.2f} ms"
    )

    print(
        f"{'C8 trust.update':20s}: "
        f"{trust_latency_ms:.2f} ms"
    )

    print(
        f"{'C8 crowd.estimate':20s}: "
        f"{crowd_latency_ms:.2f} ms"
    )

    print(
        f"\nOutput video          : "
        f"{out_path}"
    )

    print("=== End of Demo ===\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Run the full offline "
            "C1-C8 demo pipeline."
        )
    )

    parser.add_argument(
        "video_path",
        help="Path to the pre-recorded video file.",
    )

    parser.add_argument(
        "--fps",
        type=int,
        default=15,
        help=(
            "Target FPS for ingestion "
            "(default: 15)."
        ),
    )

    args = parser.parse_args()

    if not os.path.isfile(
        args.video_path
    ):
        sys.exit(
            f"Error: video file not found - "
            f"{args.video_path}"
        )

    main(
        args.video_path,
        target_fps=args.fps,
    )