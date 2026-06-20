"""Central configuration for the Realtime Anomaly Engine.

All values are read from environment variables with sane defaults so the
system runs out-of-the-box locally (against a docker-compose Kafka broker)
and is overridable in any deployment environment.
"""
from __future__ import annotations

import os

# ── Kafka ─────────────────────────────────────────────────────────────────────

KAFKA_BOOTSTRAP_SERVERS: str = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")

TOPIC_TRANSACTIONS: str = os.getenv("KAFKA_TOPIC_TRANSACTIONS", "transactions")
TOPIC_SCORED_EVENTS: str = os.getenv("KAFKA_TOPIC_SCORED", "scored-events")

KAFKA_GROUP_SCORING: str = os.getenv("KAFKA_GROUP_SCORING", "scoring-consumer")
KAFKA_GROUP_API: str = os.getenv("KAFKA_GROUP_API", "api-ws-broadcaster")

# ── Storage ───────────────────────────────────────────────────────────────────

# Avoid importing storage.database here to prevent circular imports;
# replicate the same path-join logic.
DB_PATH: str = os.getenv(
    "ANOMALY_DB_PATH",
    os.path.join(os.path.dirname(__file__), "data", "anomalies.db"),
)

# ── API / WebSocket ───────────────────────────────────────────────────────────

API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
API_PORT: int = int(os.getenv("API_PORT", "8000"))
WS_PATH: str = "/ws/anomalies"

# ── Helpers ───────────────────────────────────────────────────────────────────


def get_kafka_servers() -> str:
    """Return the Kafka bootstrap servers string (from env or default)."""
    return KAFKA_BOOTSTRAP_SERVERS


__all__ = [
    "KAFKA_BOOTSTRAP_SERVERS",
    "TOPIC_TRANSACTIONS",
    "TOPIC_SCORED_EVENTS",
    "KAFKA_GROUP_SCORING",
    "KAFKA_GROUP_API",
    "DB_PATH",
    "API_HOST",
    "API_PORT",
    "WS_PATH",
    "get_kafka_servers",
]
