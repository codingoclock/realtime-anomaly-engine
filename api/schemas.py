"""Pydantic schemas for the Anomaly API.

Schemas:
- Event: representation of original event payload (keeps common fields explicitly and allows extras)
- FeatureVector: a simple mapping of feature name -> value
- AnomalyRecord: top-level record returned by DB with fields matching `fetch_recent_anomalies`
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class Event(BaseModel):
    """Represents the original event payload stored in the DB.

    Keep common fields explicit but allow arbitrary extra fields as the event JSON
    may vary between sources.
    """

    transaction_id: Optional[str] = None
    user_id: Optional[str] = None
    amount: Optional[float] = None
    timestamp: Optional[str] = None

    class Config:
        extra = "allow"


from pydantic import RootModel


class FeatureVector(RootModel[Dict[str, Any]]):
    """Root model wrapping a mapping of feature name -> value.

    This is compatible with Pydantic v2 and can be instantiated from a plain dict.
    Access the mapping via `.root`.
    """

    root: Dict[str, Any] = Field(default_factory=dict)


class AnomalyRecord(BaseModel):
    """Schema matching the dictionaries returned by `fetch_recent_anomalies()`."""

    anomaly_row_id: int
    anomaly_score: float
    is_anomaly: bool
    processed_timestamp: str
    event: Event
    features: FeatureVector


__all__ = ["Event", "FeatureVector", "AnomalyRecord"]
