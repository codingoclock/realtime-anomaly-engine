"""Unit tests for the SQLite persistence layer."""
from __future__ import annotations

import os
import tempfile
from datetime import datetime, timedelta, timezone

from storage import database


def test_init_and_insert_and_fetch(tmp_path):
    dbfile = str(tmp_path / "test_anomalies.db")

    # init
    database.init_db(dbfile)
    assert os.path.exists(dbfile)

    # Insert a processed event
    event = {"transaction_id": "t1", "user_id": "u1", "amount": 123.45, "timestamp": datetime.now(timezone.utc).isoformat()}
    features = {"transaction_amount": 123.45}
    anomaly_id = database.insert_processed_event(event, features, anomaly_score=1.23, is_anomaly=True, db_path=dbfile)
    assert isinstance(anomaly_id, int) and anomaly_id > 0

    # Fetch recent anomalies
    results = database.fetch_recent_anomalies(db_path=dbfile, limit=10)
    assert len(results) == 1
    r = results[0]
    assert r["event"]["transaction_id"] == "t1"
    assert r["features"]["transaction_amount"] == 123.45
    assert r["is_anomaly"] is True


def test_fetch_since(tmp_path):
    dbfile = str(tmp_path / "test_anomalies2.db")
    database.init_db(dbfile)

    base = datetime.now(timezone.utc)
    # old event
    event_old = {"transaction_id": "t_old", "timestamp": (base - timedelta(hours=1)).isoformat()}
    database.insert_processed_event(event_old, {"a": 1}, anomaly_score=0.1, is_anomaly=True, db_path=dbfile, processed_ts=(base - timedelta(hours=1)).isoformat())

    # recent event
    event_new = {"transaction_id": "t_new", "timestamp": base.isoformat()}
    database.insert_processed_event(event_new, {"a": 2}, anomaly_score=2.0, is_anomaly=True, db_path=dbfile, processed_ts=base.isoformat())

    results = database.fetch_recent_anomalies(db_path=dbfile, since_ts=(base - timedelta(minutes=30)).isoformat())
    assert len(results) == 1
    assert results[0]["event"]["transaction_id"] == "t_new"
