"""Streamlit dashboard for real-time anomaly detection.

Displays recent anomalies fetched from a FastAPI backend.
Features:
- Configurable backend base URL
- Table of recent anomalies with scores and timestamps
- Simple charts: score over time and anomaly counts
- Auto-refresh via a small JS reload script
"""
from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd
import requests
import streamlit as st

# Backend configuration (cloud-safe)
API_BASE_URL = st.secrets.get(
    "API_URL",
    "http://127.0.0.1:8000"  # fallback for local development
)


# Page config
st.set_page_config(page_title="Realtime Anomaly Dashboard", layout="wide")

st.title("Realtime Anomaly Dashboard")
st.markdown("Monitoring recent anomalies and their scores")


# Sidebar: configuration
st.sidebar.header("Configuration")
default_backend = st.secrets.get("API_BASE_URL", "http://127.0.0.1:8000")
base_url = st.sidebar.text_input("Backend base URL", value=API_BASE_URL)
limit = st.sidebar.number_input("Max anomalies to fetch", min_value=1, max_value=1000, value=100, step=1)
since_ts = st.sidebar.text_input("Since timestamp (ISO8601, optional)", value="")
auto_refresh = st.sidebar.checkbox("Auto-refresh", value=True)
refresh_seconds = st.sidebar.number_input("Refresh interval (s)", min_value=2, max_value=300, value=10, step=1)


@st.cache_data(ttl=5)
def _fetch_anomalies(base: str, limit: int, since: Optional[str]) -> List[Dict[str, Any]]:
    """Fetch anomalies from backend and return parsed JSON list.

    Uses a short cache to avoid duplicate calls during a single page render.
    """ 
    url = f"{base.rstrip('/')}/anomalies"
    params: Dict[str, Any] = {"limit": int(limit)}
    if since:
        params["since_ts"] = since

    try:
        # resp = requests.get(url, params=params, timeout=5)
        resp = requests.get(f"{API_BASE_URL}/anomalies")
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as exc:  # pragma: no cover - network I/O
        st.error(f"Error fetching anomalies from {url}: {exc}")
        return []


def normalize_results(results: List[Dict[str, Any]]) -> pd.DataFrame:
    """Convert DB-shaped list of anomaly records into a flat DataFrame for display and charts."""
    if not results:
        return pd.DataFrame()

    # Build rows with selected fields
    rows: List[Dict[str, Any]] = []
    for r in results:
        processed_ts = r.get("processed_timestamp")
        # attempt to parse ISO timestamp, keep raw string if parsing fails
        try:
            ts = datetime.fromisoformat(processed_ts) if processed_ts else None
        except Exception:
            ts = None

        event = r.get("event") or {}
        features = r.get("features") or {}
        rows.append(
            {
                "anomaly_row_id": r.get("anomaly_row_id"),
                "anomaly_score": r.get("anomaly_score"),
                "is_anomaly": r.get("is_anomaly"),
                "processed_timestamp": ts,
                "transaction_id": event.get("transaction_id"),
                "user_id": event.get("user_id"),
                "transaction_amount": features.get("transaction_amount") if isinstance(features, dict) else None,
                "raw_event": event,
                "raw_features": features,
            }
        )

    df = pd.DataFrame(rows)
    if not df.empty:
        # Ensure timestamp column is datetime and sorted
        if df["processed_timestamp"].isnull().any():
            df = df.dropna(subset=["processed_timestamp"])  # drop records without valid timestamps
        df["processed_timestamp"] = pd.to_datetime(df["processed_timestamp"], utc=True)
        df = df.sort_values("processed_timestamp", ascending=False)
    return df


# Fetch data
results = _fetch_anomalies(base_url, limit, since_ts or None)
df = normalize_results(results)

col1, col2 = st.columns([3, 1])

with col1:
    st.subheader("Recent anomalies")

    if df.empty:
        st.info("No anomalies to display")
    else:
        # Display a compact table
        display_df = df[
            ["processed_timestamp", "anomaly_score", "is_anomaly", "transaction_id", "user_id", "transaction_amount"]
        ].copy()
        display_df = display_df.rename(
            columns={
                "processed_timestamp": "timestamp",
                "anomaly_score": "score",
                "transaction_amount": "amount",
            }
        )
        st.dataframe(display_df, use_container_width=True)

        st.markdown("---")

        # Score over time chart
        st.subheader("Anomaly score over time")
        score_df = df.set_index("processed_timestamp")["anomaly_score"].sort_index()
        st.line_chart(score_df)

with col2:
    st.subheader("Summary")
    st.metric("Total anomalies fetched", len(df))

    # Counts by user
    if not df.empty:
        counts = df["user_id"].fillna("<unknown>").value_counts().rename_axis("user").reset_index(name="count")
        st.table(counts)

    st.markdown("---")
    st.subheader("Charts")
    # Anomaly count over time (per minute)
    if not df.empty:
        cnt = df.set_index("processed_timestamp").resample("1Min").size()
        st.bar_chart(cnt)

# Auto-refresh via small JS snippet to reload the page every `refresh_seconds` seconds
if auto_refresh and refresh_seconds > 0:
    # Use a tiny HTML/JS snippet to trigger a reload; placed at the bottom to avoid interfering with layout
    reload_script = f"<script>setTimeout(function() {{ window.location.reload(); }}, {int(refresh_seconds) * 1000});</script>"
    st.components.v1.html(reload_script, height=0)

# Footer
st.markdown("---")
st.caption(f"Connected to backend: {base_url}")
