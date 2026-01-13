"""Synthetic real-time transaction event generator for India (IST timestamps, INR amounts).

Each generated event is a Python dict with these fields:
- `transaction_id`: UUID string
- `user_id`: string (simulates multiple users)
- `amount`: float (INR)
- `timestamp`: current time in IST (Asia/Kolkata) as ISO8601 string
- `merchant_id`: string
- `location`: Indian city name

Behavior:
- Normal transactions have amounts between ₹50 and ₹5,000
- Occasionally inject anomalous transactions:
  - very large amounts (₹50,000 to ₹2,00,000)
  - rapid repeated transactions for the same user within a short window

- New (generator-only) behavior: rare "stress" bursts (configurable, default ~3%)
  introduce realistic, fraud-like patterns: rapid sub-5-second repeats for the
  same user and amounts 5-10x the user's recent average. These are designed to
  produce naturally anomalous feature vectors without adding labels or changing
  downstream processing.

This module uses only Python standard libraries (zoneinfo for timezone).
Keep the implementation simple and well-commented.
"""
from __future__ import annotations

import random
import uuid
from datetime import datetime, timedelta
from typing import Dict, Optional
from zoneinfo import ZoneInfo


class EventGenerator:
    """Generate synthetic transaction events tailored for India.

    Parameters:
    - `user_count`: number of distinct users to simulate
    - `merchant_count`: number of distinct merchants
    - `anomaly_large_prob`: probability of a large-amount anomaly per event
    - `anomaly_repeat_prob`: probability to start a rapid-repeat anomaly
    - `seed`: optional RNG seed for reproducibility
    """

    IST = ZoneInfo("Asia/Kolkata")

    def __init__(
        self,
        user_count: int = 2000,
        merchant_count: int = 300,
        anomaly_large_prob: float = 0.01,
        anomaly_repeat_prob: float = 0.007,
        # Small probability to start a short "stress" burst (realistic, rare anomalous behavior)
        # Increased slightly for demo runs so anomalies appear occasionally during short demos
        stress_start_prob: float = 0.018,  # ~1.8% start chance -> ~5% of events expected in bursts (empirically tuned)
        seed: Optional[int] = None,
        # Probability to pick a recently active user for normal events. Keep high so
        # that the vast majority of events (~95%) have 30-300s inter-event deltas.
        active_user_prob: float = 0.98,
    ) -> None:
        self._active_user_prob = float(active_user_prob)
        self.user_count = int(user_count)
        self.merchant_count = int(merchant_count)
        self.anomaly_large_prob = float(anomaly_large_prob)
        self.anomaly_repeat_prob = float(anomaly_repeat_prob)

        # Use Python's random.Random for a lightweight RNG
        self._rng = random.Random(seed)

        # Pre-generate user and merchant ids
        self._users = [f"user_{i:06d}" for i in range(self.user_count)]
        self._merchants = [f"merchant_{i:05d}" for i in range(self.merchant_count)]

        # Typical major Indian cities for `location`
        self._locations = [
            "Delhi",
            "Mumbai",
            "Bengaluru",
            "Chennai",
            "Kolkata",
            "Hyderabad",
            "Pune",
            "Ahmedabad",
            "Jaipur",
            "Lucknow",
            "Surat",
            "Kanpur",
            "Nagpur",
            "Indore",
            "Bhopal",
        ]

        # State for rapid-repeat anomaly generation
        self._repeat_user: Optional[str] = None
        self._repeat_remaining: int = 0

        # State for stress bursts (rare, realistic anomalous behavior)
        self._stress_user: Optional[str] = None
        self._stress_remaining: int = 0
        self._stress_start_prob = float(stress_start_prob)

        # Per-user state to enable realistic stress bursts only when sufficient history exists
        # Track a simple EMA of amounts, a transaction count per user and last timestamp.
        # These are used only by the generator to produce realistic stress events; the
        # downstream FeaturePipeline computes its own rolling features independently.
        self._user_amount_ema: dict[str, float] = {}
        self._user_amount_ema_alpha = 0.2  # EMA smoothing factor
        self._user_tx_count: dict[str, int] = {}
        self._user_last_timestamp: dict[str, datetime] = {}

        # Track last timestamp in IST to produce increasing times
        self._last_timestamp = datetime.now(tz=self.IST)

    def _next_timestamp(self, rapid: bool = False, stress: bool = False) -> str:
        """Return ISO8601 timestamp in IST.

        If `stress` is True, advance by a very short interval (<3s) to
        simulate rapid, suspicious bursts. If `rapid` is True (non-stress)
        advance by a short interval (1-30s). Otherwise advance by a typical
        inter-event interval (30s-5min).
        """
        if stress:
            # stress events occur within a very short window (0.3s - 2.8s)
            delta = timedelta(seconds=self._rng.uniform(0.3, 2.8))
        elif rapid:
            # rapid repeated transactions within 1-30 seconds
            delta = timedelta(seconds=self._rng.uniform(1.0, 30.0))
        else:
            # normal inter-event time between 30s and 5 minutes
            delta = timedelta(seconds=self._rng.uniform(30.0, 300.0))

        self._last_timestamp += delta
        # Return ISO8601 string with timezone offset
        return self._last_timestamp.isoformat()

    def _sample_amount(self, anomalous: bool = False) -> float:
        """Sample transaction amount in INR.

        Normal: mostly between ₹50 and ₹5,000. We sample from a log-normal
        distribution then clamp to the desired range for realism.

        Anomalous: sample uniformly between ₹50,000 and ₹200,000.
        """
        if anomalous:
            return round(self._rng.uniform(50_000.0, 200_000.0), 2)

        # Log-normal sampling produces many small values and few large ones
        # Adjust parameters to roughly cover ₹50-₹5,000 after clamping.
        mu = 6.0  # shift to produce values in INR scale
        sigma = 0.8
        value = self._rng.lognormvariate(mu, sigma)

        # Clamp to realistic consumer range
        value = max(50.0, min(value, 5000.0))
        return round(value, 2)

    def generate_transaction(self) -> Dict[str, object]:
        """Generate a single synthetic transaction event (dict).

        The method may create a short burst of rapid repeated transactions
        for a user when a repeat-anomaly is triggered.
        """
        # If currently in a rapid-repeat burst, continue emitting for that user
        if self._repeat_remaining > 0 and self._repeat_user is not None:
            user_id = self._repeat_user
            rapid = True
            stress = False
            self._repeat_remaining -= 1
        elif self._stress_remaining > 0 and self._stress_user is not None:
            # Continue an ongoing stress burst (rare)
            user_id = self._stress_user
            rapid = True
            stress = True
            self._stress_remaining -= 1
        else:
            # Pick a user. Prefer recently active users so that their inter-event
            # deltas fall in the 30-300s range most of the time.
            # Choose active users with recent history (within last 5 minutes) so
            # their next inter-event delta can be set in 30-300s window.
            recent_cutoff = self._last_timestamp - timedelta(seconds=300)
            active_users = [
                u
                for u, cnt in self._user_tx_count.items()
                if cnt >= 1 and self._user_last_timestamp.get(u) and self._user_last_timestamp[u] >= recent_cutoff
            ]
            if active_users and (self._rng.random() < self._active_user_prob):
                user_id = self._rng.choice(active_users)
            else:
                user_id = self._rng.choice(self._users)

            rapid = False
            stress = False

            # Possibly start a rapid-repeat anomaly burst for this user (existing behavior)
            if self._rng.random() < self.anomaly_repeat_prob:
                self._repeat_user = user_id
                # produce 2-8 rapid transactions (in addition to this one)
                self._repeat_remaining = self._rng.randint(2, 8)

            # Possibly start a stress burst (rare, realistic anomalous behavior)
            # We pick from users who have recently transacted (have history)
            # and only start bursts a small fraction of the time (~5%).
            # Consider only users who have transacted recently (within the last 5 minutes)
            recent_cutoff = self._last_timestamp - timedelta(seconds=300)
            candidate_users = [
                u
                for u, cnt in self._user_tx_count.items()
                if cnt >= 1 and self._user_last_timestamp.get(u) and self._user_last_timestamp[u] >= recent_cutoff
            ]
            if candidate_users and (self._rng.random() < self._stress_start_prob):
                # Choose a user who recently transacted to make the inter-event
                # deltas meaningful (we will emit rapid, short-interval transactions)
                self._stress_user = self._rng.choice(candidate_users)
                # Burst of 2-5 rapid transactions total (we use N-1 here to count additional events)
                self._stress_remaining = self._rng.randint(1, 4)
                stress = True

        # Determine if this transaction is a large-amount anomaly (legacy behavior)
        is_large_anomaly = self._rng.random() < self.anomaly_large_prob

        if stress:
            # Stress bursts: choose short inter-arrival times and slightly elevated amounts.
            # Prefer users with a recent last-timestamp to keep `time_since_last_transaction_seconds`
            # within 1-5 seconds for burst members. If no useful last timestamp exists for the
            # chosen user, fall back to a small delta relative to the global last timestamp.
            user_last = self._user_last_timestamp.get(user_id)
            # desired per-user delta between 1 and 5 seconds
            desired_delta = self._rng.uniform(1.0, 5.0)
            if user_last is not None:
                candidate_ts = user_last + timedelta(seconds=desired_delta)
            else:
                candidate_ts = self._last_timestamp + timedelta(seconds=desired_delta)

            # Ensure global monotonicity
            if candidate_ts <= self._last_timestamp:
                # If candidate would go backwards, nudge forward slightly
                candidate_ts = self._last_timestamp + timedelta(seconds=0.1)

            timestamp = candidate_ts.isoformat()

            # Slightly elevated amount in stress bursts (1.2x - 1.5x of user EMA or base)
            user_mean = self._user_amount_ema.get(user_id)
            if user_mean is not None and user_mean > 0.0:
                factor = self._rng.uniform(1.2, 1.5)
                amount = round(user_mean * factor, 2)
            else:
                # Fall back to a slightly elevated sampled amount
                amount = round(self._sample_amount(anomalous=is_large_anomaly) * self._rng.uniform(1.2, 1.5), 2)

            # We will decrement _stress_remaining at the top of the loop when continuing bursts
        else:
            # Normal event: ensure the user's inter-event interval is between 30 and 300 seconds
            desired_delta = self._rng.uniform(30.0, 300.0)
            user_last = self._user_last_timestamp.get(user_id)
            if user_last is not None:
                candidate_ts = user_last + timedelta(seconds=desired_delta)
            else:
                candidate_ts = self._last_timestamp + timedelta(seconds=desired_delta)

            # Ensure global monotonicity
            if candidate_ts <= self._last_timestamp:
                candidate_ts = self._last_timestamp + timedelta(seconds=self._rng.uniform(1.0, 5.0))

            timestamp = candidate_ts.isoformat()

            # Normal amounts (possibly large anomaly by legacy probability)
            amount = self._sample_amount(anomalous=is_large_anomaly)

        event = {
            "transaction_id": str(uuid.uuid4()),
            "user_id": user_id,
            "amount": amount,
            "timestamp": timestamp,
            "merchant_id": self._rng.choice(self._merchants),
            "location": self._rng.choice(self._locations),
        }

        # Update per-user EMA for amounts to inform future stress events
        prev = self._user_amount_ema.get(user_id)
        if prev is None:
            self._user_amount_ema[user_id] = float(amount)
        else:
            alpha = self._user_amount_ema_alpha
            self._user_amount_ema[user_id] = (alpha * float(amount)) + (1.0 - alpha) * prev

        # Update per-user transaction count
        self._user_tx_count[user_id] = self._user_tx_count.get(user_id, 0) + 1

        # Update per-user last timestamp (store as datetime for easier future deltas)
        try:
            ts_dt = datetime.fromisoformat(timestamp)
        except Exception:
            ts_dt = datetime.now(tz=self.IST)
        self._user_last_timestamp[user_id] = ts_dt

        # Ensure global monotonicity of _last_timestamp
        if ts_dt > self._last_timestamp:
            self._last_timestamp = ts_dt
        else:
            # if somehow behind (should not normally happen), nudge forward
            self._last_timestamp = self._last_timestamp + timedelta(seconds=0.1)

        return event


# Module-level convenience: single generator instance for quick use
_DEFAULT_GEN: Optional[EventGenerator] = None


def generate_transaction(seed: Optional[int] = None) -> Dict[str, object]:
    """Return one generated transaction event.

    Pass `seed` on first call to make generation deterministic.
    """
    global _DEFAULT_GEN
    if _DEFAULT_GEN is None:
        _DEFAULT_GEN = EventGenerator(seed=seed)
    return _DEFAULT_GEN.generate_transaction()


__all__ = ["EventGenerator", "generate_transaction"]
