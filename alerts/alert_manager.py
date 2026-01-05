"""Alert manager for real-time anomaly detection.

This module provides a simple `AlertManager` class that supports:
- Score-based alerts: trigger when anomaly score crosses a threshold
- Rate-based alerts: trigger when number of anomalies exceeds a limit in a rolling time window

Design choices:
- Rate-based alerts maintain simple in-memory deques of timestamps to support efficient eviction
- Supports both global and per-user rate windows (configurable `rate_scope`)
- Alerts are printed to the console; no external side effects are performed here

Usage example:
    am = AlertManager(score_threshold=5.0, score_trigger='above', rate_limit=10, rate_window_seconds=60)
    am.check_and_alert(event, anomaly_score=6.1, is_anomaly=True)

"""
from __future__ import annotations

from collections import deque, defaultdict
from datetime import datetime, timedelta, timezone
from typing import Deque, Dict, Optional


class AlertManager:
    """Manage simple score- and rate-based alerts.

    Parameters:
    - score_threshold: numeric threshold for score-based alerts
    - score_trigger: 'above' or 'below' indicates whether to alert when
      anomaly_score is >= threshold ('above') or <= threshold ('below')
      (default: 'above')
    - rate_limit: the maximum allowed number of anomalies in the window
    - rate_window_seconds: rolling window length in seconds for rate-based alerts
    - rate_scope: 'global' or 'user' - whether to count anomalies globally or per user

    Note on anomaly score interpretation:
    - Anomaly detectors may define score directions differently. In this
      system `anomaly_score` is intended to be larger for more anomalous
      points; choose `score_trigger='above'` to alert when anomalous values
      exceed the threshold. If using a different convention, set
      `score_trigger='below'`.
    """

    def __init__(
        self,
        score_threshold: Optional[float] = None,
        score_trigger: str = "above",
        rate_limit: int = 10,
        rate_window_seconds: int = 60,
        rate_scope: str = "global",
    ) -> None:
        if score_trigger not in ("above", "below"):
            raise ValueError("score_trigger must be 'above' or 'below'")
        if rate_scope not in ("global", "user"):
            raise ValueError("rate_scope must be 'global' or 'user'")

        self.score_threshold = score_threshold
        self.score_trigger = score_trigger

        self.rate_limit = int(rate_limit)
        self.rate_window = timedelta(seconds=int(rate_window_seconds))
        self.rate_scope = rate_scope

        # State for rate-based alerts
        # Global scope: single deque of datetimes
        self._global_anomalies: Deque[datetime] = deque()
        # User scope: mapping user_id -> deque of datetimes
        self._user_anomalies: Dict[str, Deque[datetime]] = defaultdict(deque)

    # -- Helpers ---------------------------------------------------------
    def _parse_timestamp(self, ts: Optional[str]) -> datetime:
        """Parse ISO8601 timestamp string or use current time (UTC) if None.

        Ensures returned datetime is timezone-aware in UTC.
        """
        if ts is None:
            return datetime.now(timezone.utc)
        try:
            # Parse ISO8601; if naive assume UTC, otherwise convert to UTC
            dt = datetime.fromisoformat(ts)
            if dt.tzinfo is None:
                return dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc)
        except Exception:
            # Fall back to current time in UTC to avoid breaking the system if timestamp is malformed
            return datetime.now(timezone.utc)

    def _evict_old(self, dq: Deque[datetime], now: datetime) -> None:
        cutoff = now - self.rate_window
        while dq and dq[0] < cutoff:
            dq.popleft()

    # -- Alert checks ---------------------------------------------------
    def _check_score_alert(self, anomaly_score: float) -> bool:
        """Return True if score-based rule triggers an alert."""
        if self.score_threshold is None:
            return False
        if self.score_trigger == "above":
            return anomaly_score >= self.score_threshold
        else:
            return anomaly_score <= self.score_threshold

    def _check_rate_alert_global(self, now: datetime) -> bool:
        self._evict_old(self._global_anomalies, now)
        return len(self._global_anomalies) > self.rate_limit

    def _check_rate_alert_user(self, user_id: str, now: datetime) -> bool:
        dq = self._user_anomalies[user_id]
        self._evict_old(dq, now)
        return len(dq) > self.rate_limit

    # -- Public API -----------------------------------------------------
    def check_and_alert(self, event: Dict[str, object], anomaly_score: float, is_anomaly: bool) -> None:
        """Check incoming event for alert conditions and print alerts.

        - `event` is the original event dict (may contain `user_id` and `timestamp`)
        - `anomaly_score` is a numeric score (higher = more anomalous by convention)
        - `is_anomaly` is a boolean decision from the model

        This method performs the checks and prints alerts to stdout.
        """
        now = self._parse_timestamp(event.get("timestamp"))
        user_id = event.get("user_id")

        # Score-based alert
        if self._check_score_alert(anomaly_score):
            print("[ALERT][SCORE]", f"time={now.isoformat()} score={anomaly_score:.6f}", f"event={event}")

        # Rate-based alert: only track events flagged as anomalies
        if is_anomaly:
            if self.rate_scope == "global":
                self._global_anomalies.append(now)
                triggered = self._check_rate_alert_global(now)
                if triggered:
                    print(
                        "[ALERT][RATE][GLOBAL]",
                        f"time={now.isoformat()} count={len(self._global_anomalies)} window_s={self.rate_window.total_seconds()}",
                        f"event={event}",
                    )
            else:
                # user scope
                uid = str(user_id) if user_id is not None else "<unknown>"
                dq = self._user_anomalies[uid]
                dq.append(now)
                triggered = self._check_rate_alert_user(uid, now)
                if triggered:
                    print(
                        "[ALERT][RATE][USER]",
                        f"time={now.isoformat()} user={uid} count={len(dq)} window_s={self.rate_window.total_seconds()}",
                        f"event={event}",
                    )

    def reset(self) -> None:
        """Reset all in-memory alert state (useful for tests)."""
        self._global_anomalies.clear()
        self._user_anomalies.clear()


__all__ = ["AlertManager"]
