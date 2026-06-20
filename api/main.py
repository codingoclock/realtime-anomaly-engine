"""FastAPI application — Realtime Anomaly Engine.

Endpoints:
  GET  /          — health check
  GET  /anomalies — paginated historical anomalies from SQLite
  WS   /ws/anomalies — live push stream of scored transaction events

On startup an aiokafka background task subscribes to the `scored-events`
Kafka topic and broadcasts every message to all connected WebSocket clients.
If Kafka is unavailable the REST endpoints still serve normally.
"""
from __future__ import annotations

import asyncio
import json
import logging
import sys
import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Path bootstrap so the module runs directly AND as uvicorn target
try:
    from storage import database
    import config
    from api.websocket import manager
except Exception:
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    from storage import database
    import config
    from api.websocket import manager

logger = logging.getLogger("anomaly_api")


# ── aiokafka background consumer ─────────────────────────────────────────────

async def _consume_and_broadcast() -> None:
    """Read scored-events from Kafka and push each to all WS clients."""
    from aiokafka import AIOKafkaConsumer  # type: ignore[import]

    consumer = AIOKafkaConsumer(
        config.TOPIC_SCORED_EVENTS,
        bootstrap_servers=config.KAFKA_BOOTSTRAP_SERVERS,
        group_id=config.KAFKA_GROUP_API,
        value_deserializer=lambda b: json.loads(b.decode("utf-8")),
        auto_offset_reset="latest",
        enable_auto_commit=True,
    )
    await consumer.start()
    logger.info("aiokafka consumer started on topic '%s'", config.TOPIC_SCORED_EVENTS)
    try:
        async for msg in consumer:
            await manager.broadcast(msg.value)
    finally:
        await consumer.stop()


# ── Application lifespan ──────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Start the Kafka broadcaster on startup; cancel it on shutdown."""
    task: Optional[asyncio.Task] = None
    try:
        task = asyncio.create_task(_consume_and_broadcast())
        logger.info("Kafka WS broadcaster task started.")
    except Exception as exc:
        logger.warning("Kafka broadcaster could not start (REST-only mode): %s", exc)

    yield  # application runs here

    if task is not None:
        task.cancel()
        try:
            await task
        except (asyncio.CancelledError, Exception):
            pass
    logger.info("Kafka WS broadcaster stopped.")


# ── FastAPI app ───────────────────────────────────────────────────────────────

app = FastAPI(title="Realtime Anomaly API", lifespan=lifespan)

# CORS — allow the React frontend (any origin in dev; tighten in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── REST endpoints ────────────────────────────────────────────────────────────

@app.get("/", tags=["health"])
async def root() -> JSONResponse:
    return JSONResponse({"status": "ok", "message": "Anomaly API is running"})


@app.get("/anomalies", tags=["anomalies"])
async def get_anomalies(
    limit: int = Query(50, ge=1, le=1000),
    since_ts: Optional[str] = Query(None, description="ISO8601 timestamp filter"),
) -> List[dict]:
    since_iso: Optional[str] = None
    if since_ts:
        try:
            dt = datetime.fromisoformat(since_ts)
        except Exception:
            raise HTTPException(status_code=400, detail="`since_ts` must be a valid ISO8601 timestamp")
        dt = dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt.astimezone(timezone.utc)
        since_iso = dt.isoformat()

    try:
        results = database.fetch_recent_anomalies(
            db_path=config.DB_PATH,
            since_ts=since_iso,
            limit=limit,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Database error: {exc}")

    return results


# ── WebSocket endpoint ────────────────────────────────────────────────────────

@app.websocket("/ws/anomalies")
async def ws_anomalies(ws: WebSocket) -> None:
    """Live stream of scored transaction events.

    The client should connect and listen; send any text frame as a keepalive
    ping. The server echoes nothing — it only broadcasts inbound Kafka events.
    """
    await manager.connect(ws)
    try:
        while True:
            # Block until the client sends something (keepalive) or disconnects
            await ws.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(ws)
    except Exception:
        manager.disconnect(ws)


# ── Dev entrypoint ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api.main:app", host=config.API_HOST, port=config.API_PORT, log_level="info")
