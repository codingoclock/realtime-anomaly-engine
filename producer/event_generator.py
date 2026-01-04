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
        seed: Optional[int] = None,
    ) -> None:
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

        # Track last timestamp in IST to produce increasing times
        self._last_timestamp = datetime.now(tz=self.IST)

    def _next_timestamp(self, rapid: bool = False) -> str:
        """Return ISO8601 timestamp in IST.

        If `rapid` is True, advance time by a short interval (seconds).
        Otherwise advance by a typical inter-event interval (tens of seconds).
        """
        if rapid:
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
            self._repeat_remaining -= 1
        else:
            # Pick a fresh user
            user_id = self._rng.choice(self._users)
            rapid = False

            # Possibly start a rapid-repeat anomaly burst for this user
            if self._rng.random() < self.anomaly_repeat_prob:
                self._repeat_user = user_id
                # produce 2-8 rapid transactions (in addition to this one)
                self._repeat_remaining = self._rng.randint(2, 8)

        # Determine if this transaction is a large-amount anomaly
        is_large_anomaly = self._rng.random() < self.anomaly_large_prob

        amount = self._sample_amount(anomalous=is_large_anomaly)

        event = {
            "transaction_id": str(uuid.uuid4()),
            "user_id": user_id,
            "amount": amount,
            "timestamp": self._next_timestamp(rapid=rapid),
            "merchant_id": self._rng.choice(self._merchants),
            "location": self._rng.choice(self._locations),
        }

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
