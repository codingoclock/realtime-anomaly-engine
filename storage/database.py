"""SQLite persistence helpers for anomaly detection pipeline.

This module provides a small SQLite-backed layer to persist raw events,
computed features, and anomaly results. It uses the standard library's
`sqlite3` module and stores JSON payloads as TEXT for flexibility.

Schemas (simple):
- raw_events(id, transaction_id, event_json, timestamp)
- features(id, raw_event_id, features_json, timestamp)
- anomalies(id, raw_event_id, anomaly_score, is_anomaly, processed_timestamp)

Helper functions:
- init_db(db_path): create database and tables if missing
- insert_processed_event(db_path, event, features, anomaly_score, is_anomaly, processed_ts): insert all related rows
- fetch_recent_anomalies(db_path, since_ts=None, limit=100): retrieve recent anomalies

Keep the module small and dependency-free (stdlib only).
"""
from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timezone
from typing import Dict, List, Optional


DEFAULT_DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "anomalies.db")


def _ensure_dir_for_db(path: str) -> None:
    dirpath = os.path.dirname(path)
    if dirpath and not os.path.exists(dirpath):
        os.makedirs(dirpath, exist_ok=True)


def init_db(db_path: Optional[str] = None) -> None:
    """Initialize the SQLite database and required tables.

    If the database file already exists, this function will ensure tables
    exist (no-op for already-created tables).
    """
    path = db_path or DEFAULT_DB_PATH
    _ensure_dir_for_db(path)

    with sqlite3.connect(path) as conn:
        cur = conn.cursor()
        # Raw events table: store original event JSON and transaction id
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS raw_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                transaction_id TEXT,
                event_json TEXT NOT NULL,
                timestamp TEXT
            )
            """
        )

        # Features table: link to raw event
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS features (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                raw_event_id INTEGER NOT NULL,
                features_json TEXT NOT NULL,
                timestamp TEXT,
                FOREIGN KEY(raw_event_id) REFERENCES raw_events(id)
            )
            """
        )

        # Anomalies table: store score and boolean decision
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS anomalies (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                raw_event_id INTEGER NOT NULL,
                anomaly_score REAL,
                is_anomaly INTEGER,
                processed_timestamp TEXT,
                FOREIGN KEY(raw_event_id) REFERENCES raw_events(id)
            )
            """
        )

        conn.commit()


def insert_processed_event(
    event: Dict[str, object],
    features: Dict[str, object],
    anomaly_score: float,
    is_anomaly: bool,
    db_path: Optional[str] = None,
    processed_ts: Optional[str] = None,
) -> int:
    """Insert a processed event into the DB and return anomaly row id.

    The function inserts rows into `raw_events`, `features`, and `anomalies`
    in a single transaction. Returns the inserted `anomalies` row id.
    """
    path = db_path or DEFAULT_DB_PATH
    _ensure_dir_for_db(path)
    processed_ts = processed_ts or datetime.now(timezone.utc).isoformat()

    with sqlite3.connect(path) as conn:
        cur = conn.cursor()
        # Insert raw event
        cur.execute(
            "INSERT INTO raw_events (transaction_id, event_json, timestamp) VALUES (?, ?, ?)",
            (str(event.get("transaction_id")), json.dumps(event), event.get("timestamp")),
        )
        raw_id = cur.lastrowid

        # Insert features
        cur.execute(
            "INSERT INTO features (raw_event_id, features_json, timestamp) VALUES (?, ?, ?)",
            (raw_id, json.dumps(features), processed_ts),
        )

        # Insert anomaly
        cur.execute(
            "INSERT INTO anomalies (raw_event_id, anomaly_score, is_anomaly, processed_timestamp) VALUES (?, ?, ?, ?)",
            (raw_id, float(anomaly_score), int(bool(is_anomaly)), processed_ts),
        )
        anomaly_id = cur.lastrowid

        conn.commit()
    return anomaly_id


def fetch_recent_anomalies(db_path: Optional[str] = None, since_ts: Optional[str] = None, limit: int = 100) -> List[Dict[str, object]]:
    """Fetch recent anomalies joined with raw event and features.

    Args:
        since_ts: Optional ISO timestamp string; if provided only return anomalies with processed_timestamp >= since_ts
        limit: max number of rows to return (ordered by processed_timestamp desc)
    Returns:
        List of dicts with keys: anomaly_row_id, anomaly_score, is_anomaly, processed_timestamp, event (dict), features (dict)
    """
    path = db_path or DEFAULT_DB_PATH
    if not os.path.exists(path):
        return []

    q = """
    SELECT a.id, a.anomaly_score, a.is_anomaly, a.processed_timestamp,
           r.event_json, f.features_json
    FROM anomalies a
    JOIN raw_events r ON a.raw_event_id = r.id
    JOIN features f ON f.raw_event_id = r.id
    """
    params = []
    if since_ts:
        q += " WHERE a.processed_timestamp >= ?"
        params.append(since_ts)
    q += " ORDER BY a.processed_timestamp DESC LIMIT ?"
    params.append(int(limit))

    results = []
    with sqlite3.connect(path) as conn:
        cur = conn.cursor()
        cur.execute(q, params)
        for row in cur.fetchall():
            anomaly_id, score, is_anom, processed_ts, event_json, features_json = row
            try:
                event = json.loads(event_json)
            except Exception:
                event = {"raw": event_json}
            try:
                features = json.loads(features_json)
            except Exception:
                features = {"raw": features_json}
            results.append(
                {
                    "anomaly_row_id": anomaly_id,
                    "anomaly_score": float(score),
                    "is_anomaly": bool(is_anom),
                    "processed_timestamp": processed_ts,
                    "event": event,
                    "features": features,
                }
            )
    return results


__all__ = ["init_db", "insert_processed_event", "fetch_recent_anomalies", "DEFAULT_DB_PATH"]
