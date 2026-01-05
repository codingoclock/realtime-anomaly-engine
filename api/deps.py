"""Shared dependencies for the Anomaly API.

Keep this module minimal: centralize DB path configuration and provide
thin wrappers to access the persistence helpers in `storage.database`.
"""
from __future__ import annotations

import os
from functools import lru_cache
from typing import Dict, List, Optional

from storage import database


@lru_cache(maxsize=1)
def get_db_path() -> str:
    """Return the database path for the application.

    Reads the `ANOMALY_DB_PATH` environment variable if present; otherwise
    falls back to `storage.database.DEFAULT_DB_PATH`.
    """
    return os.environ.get("ANOMALY_DB_PATH", database.DEFAULT_DB_PATH)


def init_db(path: Optional[str] = None) -> None:
    """Initialize the database at `path` (or the configured path if None)."""
    database.init_db(path or get_db_path())


def fetch_recent_anomalies(since_ts: Optional[str] = None, limit: int = 50) -> List[Dict[str, object]]:
    """Fetch recent anomalies using the configured DB path.

    This is a thin wrapper around `storage.database.fetch_recent_anomalies` to
    keep the application code concise and testable.
    """
    return database.fetch_recent_anomalies(db_path=get_db_path(), since_ts=since_ts, limit=limit)


__all__ = ["get_db_path", "init_db", "fetch_recent_anomalies"]
