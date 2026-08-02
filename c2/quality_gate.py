import os
import sys
import time
from typing import Dict, Any

import cv2
import numpy as np

# Import existing quality gate implementations
from .c2_quality_brisque import BRISQUEQualityGate
from .c2_quality_piqe import PIQEQualityGate


class QualityGate:
    """Unified quality gate wrapper supporting BRISQUE and PIQE metrics.

    Args:
        method: "brisque" or "piqe" (default "brisque").
        threshold: Score threshold above which a frame is considered poor quality.
            For both methods, *higher* scores indicate worse quality.
        enhancement_method: Currently only "clahe" is supported for the HW tier.
    """

    def __init__(self, method: str = "brisque", threshold: float = None, enhancement_method: str = "clahe"):
        method = method.lower()
        if method not in {"brisque", "piqe"}:
            raise ValueError(f"Unsupported quality method: {method}. Choose 'brisque' or 'piqe'.")
        self.method = method
        self.enhancement_method = enhancement_method.lower()
        if self.enhancement_method != "clahe":
            raise ValueError("Only CLAHE enhancement is supported at the moment.")

        # Default thresholds based on the conventions used in the original scripts
        if threshold is None:
            self.threshold = 40.0 if method == "brisque" else 35.0
        else:
            self.threshold = float(threshold)

        # Instantiate model objects once (BRISQUE needs a model instance, PIQE does not).
        if self.method == "brisque":
            self._brisque = BRISQUEQualityGate()
        else:
            self._brisque = None
        # No persistent state needed for PIQE.
        self.last_latency_ms: float = 0.0

    def _enhance_clahe(self, frame: np.ndarray) -> np.ndarray:
        """Apply CLAHE contrast limited adaptive histogram equalization.

        CLAHE works on the luminance channel. The input is assumed to be a BGR uint8 image.
        """
        if frame.dtype != np.uint8:
            frame = frame.astype(np.uint8)
        # Convert to YCrCb, apply CLAHE on the Y channel, then merge back.
        ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
        y, cr, cb = cv2.split(ycrcb)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        y_eq = clahe.apply(y)
        enhanced = cv2.merge((y_eq, cr, cb))
        return cv2.cvtColor(enhanced, cv2.COLOR_YCrCb2BGR)

    def process(self, frame: np.ndarray) -> Dict[str, Any]:
        """Assess a frame and optionally enhance it.

        Returns a dict with keys:
            "frame": np.ndarray – original or enhanced frame
            "quality_score": float – metric score (higher = worse)
            "enhanced": bool – whether CLAHE was applied
            "latency_ms": float – processing time for scoring and optional enhancement
        """
        start = time.perf_counter()
        if self.method == "brisque":
            # Use the shared BRISQUE model for scoring.
            # Re‑use the logic from BRISQUEQualityGate but avoid re‑instantiating.
            # Convert to required format.
            if frame.ndim != 3 or frame.shape[2] != 3:
                raise ValueError("BRISQUE expects a 3‑channel BGR image.")
            if frame.dtype != np.uint8:
                frame_uint8 = frame.astype(np.uint8, copy=False)
            else:
                frame_uint8 = frame
            resized = cv2.resize(frame_uint8, (256, 256), interpolation=cv2.INTER_AREA)
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            gray_image = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float64) / 255.0
            # Compute features using the pre‑created model.
            features_1 = self._brisque.calculate_brisque_features(
                gray_image, kernel_size=7, sigma=7 / 6
            )
            downscaled_image = cv2.resize(
                gray_image, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC
            )
            features_2 = self._brisque.calculate_brisque_features(
                downscaled_image, kernel_size=7, sigma=7 / 6
            )
            merged_features = np.concatenate((features_1, features_2), axis=None).ravel()
            clean_features = np.array(
                [float(np.asarray(v, dtype=np.float64).reshape(-1)[0]) for v in merged_features],
                dtype=np.float64,
            )
            raw_value = self._brisque.calculate_image_quality_score(clean_features)
            flat = np.asarray(raw_value, dtype=np.float64).reshape(-1)
            score = float(flat[0]) if flat.size else -1.0
        else:  # piqe
            score, _, _ = PIQEQualityGate._compute_score_and_blocks(frame)
        poor = score < 0.0 or score > self.threshold
        enhanced = False
        out_frame = frame
        if poor and self.enhancement_method == "clahe":
            out_frame = self._enhance_clahe(frame)
            enhanced = True
        latency_ms = (time.perf_counter() - start) * 1000.0
        # Store the most recent latency for external queries
        self.last_latency_ms = latency_ms
        return {
            "frame": out_frame,
            "quality_score": float(score),
            "enhanced": bool(enhanced),
            "latency_ms": float(latency_ms),
        }


    def get_last_latency_ms(self) -> float:
        """Return the latency (in ms) of the most recent ``process`` call."""
        return self.last_latency_ms

    if len(sys.argv) < 2:
        print("Usage: python quality_gate.py <folder_path>")
        sys.exit(1)
    folder = sys.argv[1]
    if not os.path.isdir(folder):
        print(f"Error: folder does not exist: {folder}")
        sys.exit(1)

    # Initialize both gates for comparison
    br_gate = QualityGate(method="brisque")
    pi_gate = QualityGate(method="piqe")

    total = 0
    br_poor = 0
    pi_poor = 0
    br_latency = []
    pi_latency = []

    for name in sorted(os.listdir(folder)):
        path = os.path.join(folder, name)
        if not os.path.isfile(path):
            continue
        ext = os.path.splitext(name)[1].lower()
        if ext not in {".jpg", ".jpeg", ".png", ".bmp"}:
            continue
        frame = cv2.imread(path, cv2.IMREAD_COLOR)
        if frame is None:
            print(f"Warning: unable to read {name}, skipping")
            continue
        total += 1
        br_res = br_gate.process(frame)
        pi_res = pi_gate.process(frame)
        br_latency.append(br_res["latency_ms"])
        pi_latency.append(pi_res["latency_ms"])
        if br_res["enhanced"]:
            br_poor += 1
        if pi_res["enhanced"]:
            pi_poor += 1

    if total == 0:
        print("No valid images processed.")
        sys.exit(0)

    avg_br = sum(br_latency) / len(br_latency)
    avg_pi = sum(pi_latency) / len(pi_latency)
    print(f"Total images processed: {total}")
    print(f"BRISQUE – % flagged poor: {br_poor / total * 100:.2f}% , avg latency: {avg_br:.2f} ms")
    print(f"PIQE   – % flagged poor: {pi_poor / total * 100:.2f}% , avg latency: {avg_pi:.2f} ms")
