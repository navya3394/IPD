import os
import random
import re
import sys

import cv2
import numpy as np

from c2_quality_piqe import PIQEQualityGate


def _safe_stats(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    arr = np.asarray(values, dtype=np.float64)
    return float(np.mean(arr)), float(np.std(arr))


def main() -> None:
    folder = sys.argv[1] if len(sys.argv) > 1 else "./images"
    threshold = 28.0
    max_frames = 150

    if not os.path.isdir(folder):
        print(f"Error: folder does not exist: {folder}")
        return

    # Prefer numbered frame files (frame000001.jpg ...), then take first 150.
    frame_pattern = re.compile(r"^frame(\d{6})\.(jpg|jpeg|png|bmp)$", re.IGNORECASE)
    files = []
    for name in os.listdir(folder):
        m = frame_pattern.match(name)
        if m and os.path.isfile(os.path.join(folder, name)):
            files.append((int(m.group(1)), name))
    files.sort(key=lambda x: x[0])
    selected_all = [name for _, name in files[:max_frames]]
    if len(selected_all) < max_frames:
        print(f"Warning: expected {max_frames} frames, found {len(selected_all)}.")

    first_75 = selected_all[:75]
    next_75 = selected_all[75:150]
    selected = first_75 + next_75
    random.shuffle(selected)

    if not selected:
        print(f"No frame files found in: {folder}")
        return

    gate = PIQEQualityGate()
    filename_w = max(24, min(60, max(len(n) for n in selected)))

    header = (
        f"{'Filename':<{filename_w}} | "
        f"{'PIQE':>8} | {'P':>4} | {'P ms':>8}"
    )
    sep = "-" * len(header)
    print(header)
    print(sep)

    processed = 0
    scores: list[float] = []
    poor_count = 0
    latencies: list[float] = []

    for name in selected:
        path = os.path.join(folder, name)
        frame = cv2.imread(path, cv2.IMREAD_COLOR)
        if frame is None:
            print(f"Warning: failed to load image, skipping: {name}")
            continue

        res = gate.process(frame, threshold=threshold)
        score = float(res["score"])
        poor = bool(res["poor"])
        ms = float(res["latency_ms"])
        status = "FAIL" if poor else "pass"

        print(f"{name:<{filename_w}} | {score:>8.2f} | {status:>4} | {ms:>8.2f}")

        processed += 1
        scores.append(score)
        latencies.append(ms)
        if poor:
            poor_count += 1

    print(sep)
    if processed == 0:
        print("Total images processed: 0")
        print("No valid images were processed.")
        return

    mean_score, std_score = _safe_stats(scores)
    poor_pct = (poor_count / processed) * 100.0
    fast = min(latencies) if latencies else 0.0
    slow = max(latencies) if latencies else 0.0

    print(f"Total images processed: {processed}")
    print(
        f"PIQE: mean={mean_score:.2f}, std={std_score:.2f}, "
        f"% flagged poor={poor_pct:.2f}%"
    )
    print(f"PIQE latency ms: fastest={fast:.2f}, slowest={slow:.2f}")
    print(f"Threshold used: {threshold}")


if __name__ == "__main__":
    main()
