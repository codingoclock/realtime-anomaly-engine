"""Real-time feature engineering pipeline for transaction events.

This module provides a lightweight, in-memory pipeline that accepts a single
transaction event (dict) and returns computed features. The pipeline keeps
per-user state to compute incremental features suitable for streaming
anomaly detection.

Computed features (per event):
- transaction_amount: the raw transaction amount (INR)
- rolling_mean_amount_per_user: running mean amount for the user (updated online)
- transaction_count_last_1_min: number of transactions by the same user in the last 60 seconds
- time_since_last_transaction_seconds: seconds since the previous transaction for this user (0.0 for first)

Design notes:
- Uses Python stdlib only (collections.deque) for efficient sliding-window ops
- State is stored in simple dicts keyed by `user_id`
- Method `process_event(event)` is incremental and returns a dict of features
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Deque, Dict, Optional, Tuple


@dataclass
class UserState:
    """Per-user state for incremental feature calculation."""

    # Running count and sum for rolling mean across all seen transactions
    count: int = 0
    sum_amount: float = 0.0

    # Deque of (timestamp: datetime) for events in the last window
    recent_timestamps: Deque[datetime] = field(default_factory=deque)

    # Timestamp of last transaction (datetime) if available
    last_timestamp: Optional[datetime] = None


class FeaturePipeline:
    """Feature pipeline that maintains in-memory per-user state.

    Usage:
        pipeline = FeaturePipeline(window_seconds=60)
        features = pipeline.process_event(event_dict)

    The pipeline expects `event` to include at least:
      - 'user_id': str
      - 'amount': numeric (INR)
      - 'timestamp': ISO8601 string (with or without timezone)

    Timestamps are parsed with `datetime.fromisoformat` and assumed to be
    timezone-aware if the string contains an offset. If timestamp has no
    timezone info, it is treated as naive and used as-is.
    """

    def __init__(self, window_seconds: int = 60):
        self.window = timedelta(seconds=window_seconds)
        # user_id -> UserState
        self.users: Dict[str, UserState] = {}

    def _parse_timestamp(self, ts: str) -> datetime:
        """Parse ISO8601 timestamp strings into datetime objects.

        Accepts timezone-aware or naive strings. Raises ValueError on parse errors.
        """
        # datetime.fromisoformat supports many ISO-like formats including offsets
        return datetime.fromisoformat(ts)

    def _evict_old(self, user_state: UserState, now: datetime) -> None:
        """Evict timestamps older than the sliding window from the deque."""
        cutoff = now - self.window
        rt = user_state.recent_timestamps
        while rt and rt[0] < cutoff:
            rt.popleft()

    def process_event(self, event: Dict[str, object]) -> Dict[str, float | int]:
        """Process a single event and return computed features.

        Returns a dict with these keys:
          - transaction_amount (float)
          - rolling_mean_amount_per_user (float)
          - transaction_count_last_1_min (int)
          - time_since_last_transaction_seconds (float)

        This function updates internal state for the corresponding `user_id`.
        """
        # Basic validation and extraction
        if 'user_id' not in event:
            raise KeyError("event must contain 'user_id'")
        if 'amount' not in event:
            raise KeyError("event must contain 'amount'")
        if 'timestamp' not in event:
            raise KeyError("event must contain 'timestamp'")

        user_id = str(event['user_id'])
        amount = float(event['amount'])
        ts = str(event['timestamp'])

        # Parse timestamp
        now = self._parse_timestamp(ts)

        # Get or create user state
        state = self.users.get(user_id)
        if state is None:
            state = UserState()
            self.users[user_id] = state

        # Compute time since last transaction (0.0 if first event)
        if state.last_timestamp is None:
            time_since_last = 0.0
        else:
            delta = now - state.last_timestamp
            time_since_last = delta.total_seconds()

        # Update sliding window structures
        state.recent_timestamps.append(now)
        self._evict_old(state, now)

        # Update running mean info
        state.count += 1
        state.sum_amount += amount
        # Compute rolling mean; keep full precision to avoid small rounding differences
        rolling_mean = state.sum_amount / state.count

        # Update last_timestamp
        state.last_timestamp = now

        # Build features with explicit numeric types
        features = {
            'transaction_amount': float(amount),
            'rolling_mean_amount_per_user': float(rolling_mean),
            'transaction_count_last_1_min': int(len(state.recent_timestamps)),
            'time_since_last_transaction_seconds': float(round(time_since_last, 3)),
        }

        # Final validation: no feature should be None and all values must be numeric
        for k, v in features.items():
            if v is None:
                raise AssertionError(f"Feature '{k}' is None")
            if not isinstance(v, (int, float)):
                raise AssertionError(f"Feature '{k}' is not numeric: {type(v).__name__}")

        return features


# Convenience default pipeline instance
_default_pipeline: Optional[FeaturePipeline] = None


def process_event(event: Dict[str, object]) -> Dict[str, Optional[float]]:
    """Convenience function that processes an event using a module-level
    pipeline instance. This keeps simple usage concise.

    Call with `process_event(event)` to get features.
    """
    global _default_pipeline
    if _default_pipeline is None:
        _default_pipeline = FeaturePipeline()
    return _default_pipeline.process_event(event)
