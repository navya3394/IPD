import time
from typing import Any

import cv2
import numpy as np
from brisque import BRISQUE


class BRISQUEQualityGate:
    """BRISQUE-based quality gate for BGR uint8 frames."""

    @staticmethod
    def score(frame: np.ndarray) -> float:
        """
        Compute BRISQUE score in range ~[0, 100], lower is better.
        Returns -1.0 on error.
        """
        try:
            if frame is None or not isinstance(frame, np.ndarray):
                return -1.0
            if frame.ndim != 3 or frame.shape[2] != 3:
                return -1.0
            if frame.dtype != np.uint8:
                frame = frame.astype(np.uint8, copy=False)

            resized = cv2.resize(frame, (256, 256), interpolation=cv2.INTER_AREA)
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

            model = BRISQUE(url=False)
            gray_image = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float64) / 255.0

            features_1 = model.calculate_brisque_features(
                gray_image, kernel_size=7, sigma=7 / 6
            )
            downscaled_image = cv2.resize(
                gray_image, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC
            )
            features_2 = model.calculate_brisque_features(
                downscaled_image, kernel_size=7, sigma=7 / 6
            )
            merged_features = np.concatenate((features_1, features_2), axis=None).ravel()
            clean_features = np.array(
                [
                    float(np.asarray(v, dtype=np.float64).reshape(-1)[0])
                    for v in merged_features
                ],
                dtype=np.float64,
            )
            raw_value = model.calculate_image_quality_score(clean_features)
            flat = np.asarray(raw_value, dtype=np.float64).reshape(-1)
            if flat.size == 0:
                return -1.0
            value = float(flat[0])
            if not np.isfinite(value):
                return -1.0
            return float(np.clip(value, 0.0, 100.0))
        except Exception as e:
            print(f"BRISQUE error: {e}")
            return -1.0

    @staticmethod
    def is_poor(frame: np.ndarray, threshold: float = 40.0) -> bool:
        """Return True when quality is poor for the given threshold."""
        s = BRISQUEQualityGate.score(frame)
        return s < 0.0 or s > float(threshold)

    @staticmethod
    def process(frame: np.ndarray, threshold: float = 40.0) -> dict[str, Any]:
        """Return score, poor flag, and latency in ms."""
        t0 = time.perf_counter()
        s = BRISQUEQualityGate.score(frame)
        poor = s < 0.0 or s > float(threshold)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        return {"score": float(s), "poor": bool(poor), "latency_ms": float(latency_ms)}


if __name__ == "__main__":
    frame = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
    gate = BRISQUEQualityGate()
    result = gate.process(frame, threshold=40.0)
    print("BRISQUE result:", result)
