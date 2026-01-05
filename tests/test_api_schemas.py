"""Tests for `api.schemas` to ensure they accept DB-shaped data."""
from __future__ import annotations

from datetime import datetime, timezone

from api.schemas import AnomalyRecord, Event, FeatureVector


def test_schemas_accept_db_output():
    now = datetime.now(timezone.utc).isoformat()
    event = {"transaction_id": "t1", "user_id": "u1", "amount": 123.45, "timestamp": now, "extra": "x"}
    features = {"transaction_amount": 123.45}

    data = {
        "anomaly_row_id": 1,
        "anomaly_score": 2.5,
        "is_anomaly": True,
        "processed_timestamp": now,
        "event": event,
        "features": features,
    }

    rec = AnomalyRecord(**data)

    assert rec.anomaly_row_id == 1
    assert abs(rec.anomaly_score - 2.5) < 1e-9
    assert rec.is_anomaly is True
    assert rec.event.transaction_id == "t1"
    # extras on Event should be preserved
    assert getattr(rec.event, "extra") == "x"
    # FeatureVector stores mapping under .root (Pydantic RootModel)
    assert rec.features.root == features