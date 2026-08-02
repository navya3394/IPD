import os
import sys

import cv2
import numpy as np

from c2_quality_brisque import BRISQUEQualityGate
from c2_quality_piqe import PIQEQualityGate


def _safe_stats(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    arr = np.asarray(values, dtype=np.float64)
    return float(np.mean(arr)), float(np.std(arr))


def main() -> None:
    folder = sys.argv[1] if len(sys.argv) > 1 else "./images"
    supported_exts = {".jpg", ".jpeg", ".png", ".bmp"}

    if not os.path.isdir(folder):
        print(f"Error: folder does not exist: {folder}")
        return

    entries = sorted(os.listdir(folder))
    image_files = [
        name
        for name in entries
        if os.path.isfile(os.path.join(folder, name))
        and os.path.splitext(name)[1].lower() in supported_exts
    ]

    if not image_files:
        print(f"No supported images found in: {folder}")
        return

    brisque_gate = BRISQUEQualityGate()
    piqe_gate = PIQEQualityGate()

    filename_w = max(24, min(60, max(len(n) for n in image_files)))
    header = (
        f"{'Filename':<{filename_w}} | "
        f"{'BRISQUE':>8} | {'B':>4} | "
        f"{'PIQE':>8} | {'P':>4} | "
        f"{'B ms':>8} | {'P ms':>8}"
    )
    sep = "-" * len(header)

    print(header)
    print(sep)

    processed = 0
    brisque_scores: list[float] = []
    piqe_scores: list[float] = []
    brisque_poor = 0
    piqe_poor = 0
    agree = 0
    brisque_latencies: list[float] = []
    piqe_latencies: list[float] = []

    for name in image_files:
        path = os.path.join(folder, name)
        frame = cv2.imread(path, cv2.IMREAD_COLOR)
        if frame is None:
            print(f"Warning: failed to load image, skipping: {name}")
            continue

        b_res = brisque_gate.process(frame, threshold=40.0)
        p_res = piqe_gate.process(frame, threshold=28.0)

        b_score = float(b_res["score"])
        p_score = float(p_res["score"])
        b_poor = bool(b_res["poor"])
        p_poor = bool(p_res["poor"])
        b_ms = float(b_res["latency_ms"])
        p_ms = float(p_res["latency_ms"])

        b_status = "FAIL" if b_poor else "pass"
        p_status = "FAIL" if p_poor else "pass"

        print(
            f"{name:<{filename_w}} | "
            f"{b_score:>8.2f} | {b_status:>4} | "
            f"{p_score:>8.2f} | {p_status:>4} | "
            f"{b_ms:>8.2f} | {p_ms:>8.2f}"
        )

        processed += 1
        brisque_scores.append(b_score)
        piqe_scores.append(p_score)
        brisque_latencies.append(b_ms)
        piqe_latencies.append(p_ms)
        if b_poor:
            brisque_poor += 1
        if p_poor:
            piqe_poor += 1
        if b_poor == p_poor:
            agree += 1

    print(sep)

    if processed == 0:
        print("Total images processed: 0")
        print("No valid images were processed.")
        return

    b_mean, b_std = _safe_stats(brisque_scores)
    p_mean, p_std = _safe_stats(piqe_scores)

    b_poor_pct = (brisque_poor / processed) * 100.0
    p_poor_pct = (piqe_poor / processed) * 100.0
    agree_pct = (agree / processed) * 100.0

    b_fast = min(brisque_latencies) if brisque_latencies else 0.0
    b_slow = max(brisque_latencies) if brisque_latencies else 0.0
    p_fast = min(piqe_latencies) if piqe_latencies else 0.0
    p_slow = max(piqe_latencies) if piqe_latencies else 0.0

    print(f"Total images processed: {processed}")
    print(
        f"BRISQUE: mean={b_mean:.2f}, std={b_std:.2f}, "
        f"% flagged poor={b_poor_pct:.2f}%"
    )
    print(
        f"PIQE:    mean={p_mean:.2f}, std={p_std:.2f}, "
        f"% flagged poor={p_poor_pct:.2f}%"
    )
    print(f"Agreement rate: {agree_pct:.2f}%")
    print(f"BRISQUE latency ms: fastest={b_fast:.2f}, slowest={b_slow:.2f}")
    print(f"PIQE latency ms:    fastest={p_fast:.2f}, slowest={p_slow:.2f}")


if __name__ == "__main__":
    main()
