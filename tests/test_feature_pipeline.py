"""Unit tests for the real-time feature pipeline."""
from __future__ import annotations

from datetime import datetime, timedelta

from consumer.feature_pipeline import FeaturePipeline


def _iso(ts: datetime) -> str:
    return ts.isoformat()


def test_basic_incremental_features():
    p = FeaturePipeline(window_seconds=60)

    base = datetime.now()
    e1 = {"user_id": "u1", "amount": 100.0, "timestamp": _iso(base)}
    f1 = p.process_event(e1)

    assert f1["transaction_amount"] == 100.0
    assert f1["rolling_mean_amount_per_user"] == 100.0
    assert f1["transaction_count_last_1_min"] == 1
    assert f1["time_since_last_transaction_seconds"] == 0.0

    # all features must be present and numeric
    assert all(v is not None for v in f1.values())
    assert all(isinstance(v, (int, float)) for v in f1.values())

    # second event after 30 seconds
    e2 = {"user_id": "u1", "amount": 200.0, "timestamp": _iso(base + timedelta(seconds=30))}
    f2 = p.process_event(e2)

    assert f2["transaction_amount"] == 200.0
    assert f2["rolling_mean_amount_per_user"] == 150.0
    assert f2["transaction_count_last_1_min"] == 2
    assert f2["time_since_last_transaction_seconds"] == 30.0

    assert all(v is not None and isinstance(v, (int, float)) for v in f2.values())

    # third event after 120 seconds (first event expired from 1-min window)
    e3 = {"user_id": "u1", "amount": 50.0, "timestamp": _iso(base + timedelta(seconds=150))}
    f3 = p.process_event(e3)

    assert f3["transaction_amount"] == 50.0
    assert f3["rolling_mean_amount_per_user"] == (100.0 + 200.0 + 50.0) / 3
    assert f3["transaction_count_last_1_min"] == 1
    assert f3["time_since_last_transaction_seconds"] == 120.0

    assert all(v is not None and isinstance(v, (int, float)) for v in f3.values())


def test_multiple_users_independent_state():
    p = FeaturePipeline(window_seconds=60)
    now = datetime.now()
    e1 = {"user_id": "u1", "amount": 100.0, "timestamp": _iso(now)}
    e2 = {"user_id": "u2", "amount": 300.0, "timestamp": _iso(now + timedelta(seconds=10))}

    f1 = p.process_event(e1)
    f2 = p.process_event(e2)

    assert f1["rolling_mean_amount_per_user"] == 100.0
    assert f2["rolling_mean_amount_per_user"] == 300.0
    assert f1["transaction_count_last_1_min"] == 1
    assert f2["transaction_count_last_1_min"] == 1
