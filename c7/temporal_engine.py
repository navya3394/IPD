# c7/temporal_engine.py
"""
Temporal Sequence Modeling Engine (C7)
===================================
This module sits between the Action Recognition component (C6) and the Trust
Score Engine (C8).  It receives a stream of action events, keeps a short history
per ``track_id`` and evaluates that history against a *suspicion table*.

The implementation is deliberately lightweight – no external ML models are
required – and follows the same latency‑logging convention used throughout the
project.

Matching Logic
--------------
For each incoming event we append the ``action_label`` to a per‑track deque of
size ``window_size``.  A pattern matches when **its tokens appear as a contiguous
sub‑sequence** anywhere in the current window.  This is simple, deterministic and
covers both short‑term and longer‑term behaviours (e.g. ``[linger, linger, reach]``).
If multiple patterns match we keep the one with the highest ``suspicion_score``.

Single‑action high‑severity events (``punch``, ``kick`` or ``fall``) are treated
as implicit patterns and always raise a high score, regardless of surrounding
context.
"""

from __future__ import annotations

import time
from collections import deque, defaultdict
from typing import List, Dict, Tuple, Optional

# ---------------------------------------------------------------------------
# Action taxonomy – defined once and reused throughout the code base
# ---------------------------------------------------------------------------
ACTION_LABELS = (
    "walk",
    "run",
    "linger",
    "push",
    "punch",
    "kick",
    "fall",
    "reach",
    "normal",
)

# Helper type aliases for readability
ActionEvent = Dict[str, object]
Pattern = Tuple[str, ...]
SuspicionTable = Dict[Pattern, float]


class TemporalEngine:
    """Core engine for temporal sequence modelling.

    Parameters
    ----------
    window_size: int, default 20
        Number of recent ``action_label`` tokens to retain for each ``track_id``.
    suspicion_table: dict, optional
        Mapping of ``action_label`` sequences (as a tuple) to a suspicion score
        in the range ``0.0`` – ``1.0``.  If omitted a sensible default table is
        constructed internally.
    """

    # ---------------------------------------------------------------------
    # Default suspicion table (class‑level constant for easy reference)
    # ---------------------------------------------------------------------
    _DEFAULT_TABLE: SuspicionTable = {
        # Single high‑severity actions
        ("punch",): 0.8,
        ("kick",): 0.8,
        ("fall",): 0.7,
        # Moderate‑severity sequences
        ("linger", "linger", "reach"): 0.5,
        ("walk", "linger", "reach"): 0.5,
        # Repeated pushes – handled specially (see _score_pushes())
    }

    def __init__(self, window_size: int = 20, suspicion_table: Optional[SuspicionTable] = None):
        self.window_size = window_size
        # Use a copy so the caller cannot mutate our internal defaults
        self.suspicion_table: SuspicionTable = dict(self._DEFAULT_TABLE)
        if suspicion_table:
            # Merge user‑provided entries, overriding defaults where keys clash
            self.suspicion_table.update(suspicion_table)
        # Per‑track history of action labels
        self._history: defaultdict[int, deque[str]] = defaultdict(
            lambda: deque(maxlen=self.window_size)
        )
        # Latency of the most recent ``update`` call (ms)
        self._last_latency_ms: float = 0.0

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------
    def update(self, action_event: ActionEvent) -> Dict[str, object]:
        """Process a single C6‑style event.

        The method validates the input, updates the internal token window and
        returns a dictionary containing the current suspicion assessment.
        """
        start = time.perf_counter()

        # -----------------------------------------------------------------
        # Validate required keys – raise a clear error if anything is missing
        # -----------------------------------------------------------------
        required_keys = {"track_id", "action_label", "confidence", "frame_id", "timestamp"}
        missing = required_keys - action_event.keys()
        if missing:
            raise ValueError(f"Missing required key(s) in action_event: {', '.join(sorted(missing))}")

        track_id: int = int(action_event["track_id"])
        action_label: str = str(action_event["action_label"]).lower()
        frame_id: int = int(action_event["frame_id"])
        timestamp: float = float(action_event["timestamp"])

        if action_label not in ACTION_LABELS:
            raise ValueError(f"Invalid action_label '{action_label}'. Expected one of {ACTION_LABELS}")

        # -----------------------------------------------------------------
        # Update per‑track deque
        # -----------------------------------------------------------------
        history_deque = self._history[track_id]
        history_deque.append(action_label)
        window: List[str] = list(history_deque)

        # -----------------------------------------------------------------
        # Scoring – start with a neutral baseline
        # -----------------------------------------------------------------
        suspicion_score = 0.0
        matched_pattern: Optional[str] = None

        # 1️⃣  High‑severity single actions (punch, kick, fall) – always apply
        if action_label in {"punch", "kick", "fall"}:
            # Use the explicit entry from the default table if present
            base_score = self.suspicion_table.get((action_label,), 0.7)
            suspicion_score = max(suspicion_score, base_score)
            matched_pattern = action_label

        # 2️⃣  Repeated "push" – escalation based on count within the window
        push_score, push_pattern = self._score_pushes(window)
        if push_score > suspicion_score:
            suspicion_score = push_score
            matched_pattern = push_pattern

        # 3️⃣  Pattern look‑ups from the suspicion table (excluding the single‑
        #     action entries already handled above)
        for pattern, score in self.suspicion_table.items():
            # Skip single‑action high‑severity patterns – they were processed
            if len(pattern) == 1:
                continue
            if self._pattern_in_window(pattern, window):
                if score > suspicion_score:
                    suspicion_score = score
                    matched_pattern = " → ".join(pattern)

        # -----------------------------------------------------------------
        # Assemble result payload
        # -----------------------------------------------------------------
        result = {
            "track_id": track_id,
            "suspicion_score": float(suspicion_score),
            "matched_pattern": matched_pattern,
            "window": window,
            "frame_id": frame_id,
            "timestamp": timestamp,
        }

        # Record latency for external consumers
        self._last_latency_ms = (time.perf_counter() - start) * 1000.0
        return result

    def get_last_latency_ms(self) -> float:
        """Return the latency (in ms) of the most recent ``update`` call."""
        return self._last_latency_ms

    def get_active_track_ids(self) -> List[int]:
        """Return a list of ``track_id`` values that currently have a history."""
        return list(self._history.keys())

    def reset_track(self, track_id: int) -> None:
        """Clear the stored history for *track_id*.

        This is useful when the upstream tracker (C5) drops a track after it has
        been lost for ``max_lost_frames``.
        """
        if track_id in self._history:
            del self._history[track_id]

    # ---------------------------------------------------------------------
    # Internal helper methods
    # ---------------------------------------------------------------------
    @staticmethod
    def _pattern_in_window(pattern: Pattern, window: List[str]) -> bool:
        """Check whether *pattern* appears as a contiguous subsequence in *window*.

        The implementation slides a fixed‑size view over ``window`` and performs a
        direct equality comparison.  This is O(N × M) in the worst case but fast
        enough for the very small windows (≤ 20) used here.
        """
        pat_len = len(pattern)
        if pat_len == 0 or pat_len > len(window):
            return False
        for i in range(len(window) - pat_len + 1):
            if tuple(window[i : i + pat_len]) == pattern:
                return True
        return False

    @staticmethod
    def _score_pushes(window: List[str]) -> Tuple[float, Optional[str]]:
        """Escalating score for repeated ``push`` actions.

        The score grows linearly with the count of ``push`` in the current window
        but caps at ``0.7`` to stay within the allowed range.
        """
        count = window.count("push")
        if count == 0:
            return 0.0, None
        # Base score 0.2 per push, capped at 0.7
        score = min(0.2 * count, 0.7)
        pattern = "push" * count  # e.g. "pushpush" – useful for debugging
        return score, pattern


# ---------------------------------------------------------------------------
# Demo / sanity‑check when the module is executed directly
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import random

    engine = TemporalEngine()

    # Simulated stream – three agents with IDs 1, 2, 3
    synthetic_events = [
        # Track 1 – benign walk/run sequence (low suspicion)
        {"track_id": 1, "action_label": "walk", "confidence": 0.9, "frame_id": 1, "timestamp": 0.0},
        {"track_id": 1, "action_label": "walk", "confidence": 0.92, "frame_id": 2, "timestamp": 0.1},
        {"track_id": 1, "action_label": "run", "confidence": 0.88, "frame_id": 3, "timestamp": 0.2},
        # Track 2 – suspicious linger‑linger‑reach pattern (moderate)
        {"track_id": 2, "action_label": "linger", "confidence": 0.85, "frame_id": 1, "timestamp": 0.05},
        {"track_id": 2, "action_label": "linger", "confidence": 0.80, "frame_id": 2, "timestamp": 0.15},
        {"track_id": 2, "action_label": "reach", "confidence": 0.90, "frame_id": 3, "timestamp": 0.25},
        # Track 3 – high‑severity single action (punch) – should fire high score
        {"track_id": 3, "action_label": "punch", "confidence": 0.95, "frame_id": 1, "timestamp": 0.02},
        # Additional pushes to demonstrate escalation
        {"track_id": 2, "action_label": "push", "confidence": 0.70, "frame_id": 4, "timestamp": 0.35},
        {"track_id": 2, "action_label": "push", "confidence": 0.72, "frame_id": 5, "timestamp": 0.45},
    ]

    print("--- TemporalEngine demo ---")
    for ev in synthetic_events:
        result = engine.update(ev)
        print(
            f"Track {result['track_id']}: score={result['suspicion_score']:.2f} "
            f"matched={result['matched_pattern']!r} window={result['window']}"
        )
    print("Active tracks:", engine.get_active_track_ids())
    # Reset one track and show that its history is cleared
    engine.reset_track(1)
    print("After reset, active tracks:", engine.get_active_track_ids())
