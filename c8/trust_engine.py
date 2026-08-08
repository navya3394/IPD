# c8/trust_engine.py
"""
Trust Score Engine (C8)
======================
Maintains a per-track trust score from 0.0 to 1.0.

Suspicion events reduce trust. During calm periods, trust recovers toward the
configured baseline. Crowd density only dampens the suspicion-driven drop; it
never changes the score by itself.
"""

from __future__ import annotations

import time
from typing import Any, Dict


class TrustScoreEngine:
    """Rule-based per-track trust scoring engine."""

    def __init__(
        self,
        zone_thresholds: Dict[str, float] | None = None,
        decay_rate: float = 0.05,
        baseline: float = 0.8,
    ) -> None:
        self.zone_thresholds = (
            dict(zone_thresholds)
            if zone_thresholds is not None
            else {"public": 0.3, "restricted": 0.5}
        )
        self.decay_rate = float(decay_rate)
        self.baseline = float(baseline)
        self._track_scores: Dict[int, float] = {}
        self._last_latency_ms = 0.0

    def update(
        self,
        track_id: int,
        suspicion_event: Dict[str, Any],
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Update and return the trust state for one track."""

        start = time.perf_counter()

        if "suspicion_score" not in suspicion_event:
            raise ValueError("Missing required key: 'suspicion_score'")

        if track_id not in self._track_scores:
            self._track_scores[track_id] = 1.0

        suspicion_score = float(suspicion_event["suspicion_score"])

        zone = context.get("zone", "public")
        if zone not in self.zone_thresholds:
            raise ValueError(f"Unrecognized zone: {zone!r}")

        # Crowd density can only dampen the suspicion penalty.
        max_zone_density = float(context.get("max_zone_density", 0.0))
        density_level = min(max(max_zone_density, 0.0), 1.0)

        # At maximum density, reduce the suspicion-driven penalty by 10%.
        crowd_damping = 1.0 - (0.10 * density_level)

        current = self._track_scores[track_id]

        if suspicion_score > 0.05:
            drop_factor = 0.60
            trust_drop = suspicion_score * drop_factor * crowd_damping
            new_score = current - trust_drop
        else:
            # Calm period: recover toward baseline.
            new_score = current + (self.baseline - current) * self.decay_rate

        new_score = min(max(new_score, 0.0), 1.0)
        self._track_scores[track_id] = new_score

        self._last_latency_ms = (time.perf_counter() - start) * 1000.0

        return {
            "track_id": track_id,
            "trust_score": float(new_score),
            "alert": bool(new_score < self.zone_thresholds[zone]),
            "threshold_used": float(self.zone_thresholds[zone]),
            "frame_id": suspicion_event.get("frame_id"),
            "timestamp": suspicion_event.get("timestamp"),
        }

    def get_last_latency_ms(self) -> float:
        """Return the latency of the most recent update call in milliseconds."""
        return self._last_latency_ms


if __name__ == "__main__":
    engine = TrustScoreEngine()
    track_id = 1

    print("Calm period:")
    for i in range(65):
        result = engine.update(
            track_id,
            {"suspicion_score": 0.0, "frame_id": i, "timestamp": i},
            {"zone": "public", "total_people": 10, "max_zone_density": 0.0},
        )

        if i in {0, 4, 9, 19, 39, 64}:
            print(
                f"  calm frame={i:02d} "
                f"trust={result['trust_score']:.3f} "
                f"alert={result['alert']}"
            )

    print("\nHigh-suspicion spike:")
    spike = engine.update(
        track_id,
        {"suspicion_score": 0.85, "frame_id": 65, "timestamp": 65},
        {"zone": "public", "total_people": 10, "max_zone_density": 0.0},
    )

    print(
        f"  spike frame=65 "
        f"trust={spike['trust_score']:.3f} "
        f"alert={spike['alert']}"
    )

    print("\nRecovery:")
    for i in range(1, 21):
        result = engine.update(
            track_id,
            {"suspicion_score": 0.0, "frame_id": 65 + i, "timestamp": 65 + i},
            {"zone": "public", "total_people": 10, "max_zone_density": 0.0},
        )

        if i in {1, 5, 10, 15, 20}:
            print(
                f"  calm frame={65 + i:02d} "
                f"trust={result['trust_score']:.3f} "
                f"alert={result['alert']}"
            )