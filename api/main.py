"""FastAPI application exposing endpoints for real-time anomalies.

Endpoints:
- GET / : health check
- GET /anomalies : list recent anomalies from SQLite store

This module uses the existing `storage.database` helpers.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, List, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse

from storage import database
import os

app = FastAPI(title="Realtime Anomaly API")


@app.get("/", tags=["health"])
async def root() -> JSONResponse:
    """Simple health endpoint."""
    return JSONResponse({"status": "ok", "message": "Anomaly API is running"})


@app.get("/anomalies", tags=["anomalies"])
async def get_anomalies(
    limit: int = Query(50, ge=1, le=1000, description="Max number of anomalies to return"),
    since_ts: Optional[str] = Query(None, description="Optional ISO8601 timestamp to filter anomalies from (inclusive)"),
) -> List[dict]:
    """Return recent anomalies from the database.

    - `limit`: maximum number of rows to return
    - `since_ts`: optional ISO8601 timestamp string; if provided only return anomalies with processed_timestamp >= since_ts

    The endpoint validates and normalizes `since_ts` to UTC before querying.
    """
    # Validate and normalize since_ts if provided
    since_iso: Optional[str] = None
    if since_ts:
        try:
            dt = datetime.fromisoformat(since_ts)
        except Exception:
            raise HTTPException(status_code=400, detail="`since_ts` must be a valid ISO8601 timestamp")
        # If naive, assume UTC, otherwise convert to UTC
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        else:
            dt = dt.astimezone(timezone.utc)
        since_iso = dt.isoformat()

    # Query the database
    try:
        db_path = os.environ.get("DB_PATH")
        results = database.fetch_recent_anomalies(
        db_path=db_path,
        since_ts=since_iso,
        limit=limit,
        )
    except Exception as exc:  # pragma: no cover - defensive error handling
        raise HTTPException(status_code=500, detail=f"Error querying database: {exc}")

    return results


# Optional: run with `python -m api.main` for development
if __name__ == "__main__":  # pragma: no cover - manual dev run
    import uvicorn

    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, log_level="info")
