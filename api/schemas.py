"""Pydantic schemas for the Anomaly API.

Schemas:
  Event         — original transaction event payload
  FeatureVector — mapping of feature name → numeric value
  AnomalyRecord — row returned by fetch_recent_anomalies (REST /anomalies)
  ScoredEvent   — message published to Kafka scored-events topic and
                  broadcast over WebSocket /ws/anomalies
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, RootModel


class Event(BaseModel):
    """Original transaction payload stored in raw_events."""

    transaction_id: Optional[str] = None
    user_id: Optional[str] = None
    amount: Optional[float] = None
    timestamp: Optional[str] = None

    class Config:
        extra = "allow"


class FeatureVector(RootModel[Dict[str, Any]]):
    """Mapping of feature name → value (Pydantic v2 RootModel)."""

    root: Dict[str, Any] = Field(default_factory=dict)


class AnomalyRecord(BaseModel):
    """Schema matching dictionaries returned by fetch_recent_anomalies()."""

    anomaly_row_id: int
    anomaly_score: float
    is_anomaly: bool
    processed_timestamp: str
    event: Event
    features: FeatureVector


class ScoredEvent(BaseModel):
    """Schema for messages on the scored-events Kafka topic (schema_version=1).

    Published by consumer/consumer.py via build_scored_message() and broadcast
    to React clients over WebSocket /ws/anomalies.
    """

    schema_version: int = 1
    transaction_id: str
    user_id: str
    amount: float
    timestamp: str
    processed_timestamp: str
    anomaly_score: float
    is_anomaly: bool
    explanation: List[str] = Field(default_factory=list)
    features: Dict[str, float] = Field(default_factory=dict)
    event: Dict[str, Any] = Field(default_factory=dict)
    anomaly_row_id: Optional[int] = None


__all__ = ["Event", "FeatureVector", "AnomalyRecord", "ScoredEvent"]
