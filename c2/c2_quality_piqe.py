import time
from typing import Any

import cv2
import numpy as np


class PIQEQualityGate:
    """PIQE-style quality gate for BGR uint8 frames."""

    _BLOCK_SIZE = 16
    _ACTIVE_VAR_THRESH = 0.001
    _EPS = 1e-8

    @classmethod
    def _compute_score_and_blocks(cls, frame: np.ndarray) -> tuple[float, int, int]:
        if frame is None or not isinstance(frame, np.ndarray):
            return -1.0, 0, 0
        if frame.ndim != 3 or frame.shape[2] != 3:
            return -1.0, 0, 0

        if frame.dtype != np.uint8:
            frame = frame.astype(np.uint8, copy=False)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float64) / 255.0
        h, w = gray.shape
        b = cls._BLOCK_SIZE
        h_crop = (h // b) * b
        w_crop = (w // b) * b
        if h_crop == 0 or w_crop == 0:
            return -1.0, 0, 0

        gray = gray[:h_crop, :w_crop]
        blocks = gray.reshape(h_crop // b, b, w_crop // b, b).transpose(0, 2, 1, 3)
        total_blocks = blocks.shape[0] * blocks.shape[1]

        block_vars = blocks.var(axis=(2, 3))
        active_mask = block_vars > cls._ACTIVE_VAR_THRESH
        active_blocks = int(np.count_nonzero(active_mask))

        if active_blocks == 0:
            return 0.0, 0, int(total_blocks)

        active = blocks[active_mask]
        mu = active.mean(axis=(1, 2), keepdims=True)
        sigma = np.sqrt(active.var(axis=(1, 2), keepdims=True) + cls._EPS)
        nss = (active - mu) / sigma

        nss_mean = nss.mean(axis=(1, 2))
        nss_var = nss.var(axis=(1, 2))
        abs_m = np.mean(np.abs(nss), axis=(1, 2)) + cls._EPS
        sq_m = np.mean(nss * nss, axis=(1, 2)) + cls._EPS
        ratio = (abs_m * abs_m) / sq_m

        # Approximate GGD shape deviation from Gaussian using NSS moment ratio.
        target_ratio = 2.0 / np.pi
        ggd_dev = np.abs(ratio - target_ratio) / target_ratio

        mean_dev = np.abs(nss_mean)
        var_dev = np.abs(nss_var - 1.0)

        # Blur/noise sensitivity from local gradients and variance.
        gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        grad_mag = np.sqrt(gx * gx + gy * gy)
        grad_blocks = grad_mag[:h_crop, :w_crop].reshape(
            h_crop // b, b, w_crop // b, b
        ).transpose(0, 2, 1, 3)
        active_grad = grad_blocks[active_mask]
        grad_energy = np.mean(active_grad, axis=(1, 2))

        active_var = block_vars[active_mask]
        low_texture_penalty = 1.0 / (active_var + 0.002)
        low_edge_penalty = 1.0 / (grad_energy + 0.02)

        # Weighted block distortion; scaled into [0, 100].
        block_dist = (
            22.0 * var_dev
            + 10.0 * mean_dev
            + 18.0 * ggd_dev
            + 0.55 * low_texture_penalty
            + 0.35 * low_edge_penalty
        )
        score = float(np.mean(block_dist))
        return float(np.clip(score, 0.0, 100.0)), active_blocks, int(total_blocks)

    @classmethod
    def score(cls, frame: np.ndarray) -> float:
        """
        Compute PIQE-style score in [0, 100], lower is better.
        Returns -1.0 on error.
        """
        try:
            score, _, _ = cls._compute_score_and_blocks(frame)
            return float(score)
        except Exception:
            return -1.0

    @classmethod
    def is_poor(cls, frame: np.ndarray, threshold: float = 35.0) -> bool:
        s = cls.score(frame)
        return s < 0.0 or s > float(threshold)

    @classmethod
    def process(cls, frame: np.ndarray, threshold: float = 35.0) -> dict[str, Any]:
        t0 = time.perf_counter()
        score, active_blocks, total_blocks = cls._compute_score_and_blocks(frame)
        poor = score < 0.0 or score > float(threshold)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        return {
            "score": float(score),
            "poor": bool(poor),
            "latency_ms": float(latency_ms),
            "active_blocks": int(active_blocks),
            "total_blocks": int(total_blocks),
        }


if __name__ == "__main__":
    frame = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
    blurred = cv2.GaussianBlur(frame, (0, 0), sigmaX=5, sigmaY=5)

    gate = PIQEQualityGate()
    r1 = gate.process(frame, threshold=35.0)
    r2 = gate.process(blurred, threshold=35.0)

    print("PIQE random:", r1)
    print("PIQE blurred:", r2)
    print("Blurred higher score:", r2["score"] > r1["score"])
