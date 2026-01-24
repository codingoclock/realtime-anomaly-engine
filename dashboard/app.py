"""Professional Finance-Grade Real-Time Anomaly Monitoring Console.

A production-ready Streamlit dashboard for monitoring real-time transaction
anomalies with modern dark analytics UI, real-time charts, and live event streams.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from collections import deque
import time

import pandas as pd
import numpy as np
import requests
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

# ============================================================================
# CONFIGURATION & THEME SETUP
# ============================================================================

# Backend configuration
try:
    API_BASE_URL = st.secrets.get("API_URL", "http://127.0.0.1:8000")
except (FileNotFoundError, KeyError):
    API_BASE_URL = "http://127.0.0.1:8000"

# Page config with custom theme - Dark mode default
st.set_page_config(
    page_title="Anomaly Monitoring Console",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Modern dark analytics color palette with purple accents
COLOR_NORMAL = "#10B981"      # Green
COLOR_WARNING = "#F59E0B"     # Amber
COLOR_ANOMALY = "#EF4444"     # Red
COLOR_NEUTRAL = "#6B7280"     # Gray
COLOR_PRIMARY_PURPLE = "#A78BFA"   # Purple (shadcn-inspired)
COLOR_DARK_PURPLE = "#7C3AED"      # Darker purple for highlights

THEMES = {
    "dark": {
        "bg_primary": "#0F172A",
        "bg_secondary": "#1E293B",
        "bg_tertiary": "#334155",
        "text_primary": "#F1F5F9",
        "text_secondary": "#94A3B8",
        "border": "#475569",
        "chart_template": "plotly_dark",
        "purple_accent": COLOR_PRIMARY_PURPLE,
        "purple_dark": COLOR_DARK_PURPLE
    }
}

# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================

if "theme" not in st.session_state:
    st.session_state.theme = "dark"

if "live_transactions" not in st.session_state:
    st.session_state.live_transactions = deque(maxlen=50)

if "transaction_stats" not in st.session_state:
    st.session_state.transaction_stats = {"total": 0, "anomalies": 0}

if "time_series_buffer" not in st.session_state:
    # Store time-series for charts (last 60 minutes of minute-level data)
    st.session_state.time_series_buffer = deque(maxlen=60)

if "anomalies_per_minute" not in st.session_state:
    st.session_state.anomalies_per_minute = {}

if "backend_error" not in st.session_state:
    st.session_state.backend_error = None

if "last_update" not in st.session_state:
    st.session_state.last_update = None

# ============================================================================
# GHOST DATA GENERATORS (UI-ONLY - Never Logged or Persisted)
# ============================================================================

def ghost_time_index(num_points: int = 30) -> list[str]:
    """Generate baseline timestamps for ghost data.
    
    Deterministic time index for visual consistency.
    """
    now = datetime.now()
    times = []
    for i in range(num_points - 1, -1, -1):
        t = now - timedelta(seconds=i * 2)
        times.append(t.strftime("%H:%M:%S"))
    return times

def ghost_flat_series(num_points: int = 30, value: float = 0.35) -> list[float]:
    """Generate flat baseline series for visual scaffolding."""
    return [value] * num_points

def ghost_wave_series(num_points: int = 30, amplitude: float = 0.1, center: float = 0.35) -> list[float]:
    """Generate smooth wave for baseline visual interest."""
    import math
    series = []
    for i in range(num_points):
        angle = (i / num_points) * 2 * math.pi
        noise = amplitude * math.sin(angle)
        series.append(max(0, min(1, center + noise)))
    return series

def ghost_transaction_rows(num_rows: int = 8) -> list[dict]:
    """Generate placeholder transaction rows.
    
    Muted, clearly placeholder data for table scaffolding.
    """
    rows = []
    base_time = datetime.now()
    
    amounts = [1250.00, 875.50, 2340.00, 450.25, 1895.75, 3200.00, 625.50, 1440.00]
    users = ["USR-001", "USR-042", "USR-083", "USR-015", "USR-067", "USR-099", "USR-021", "USR-054"]
    
    for i in range(min(num_rows, len(amounts))):
        t = base_time - timedelta(seconds=i * 3)
        rows.append({
            "timestamp": t.strftime("%H:%M:%S"),
            "user_id": users[i],
            "amount": amounts[i],
            "anomaly_score": 0.25 + (i * 0.05),
            "is_anomaly": False,
            "explanation": "Baseline data",
            "_is_ghost": True
        })
    return rows

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_risk_level(anomaly_rate: float) -> tuple[str, str]:
    """Determine risk level based on anomaly rate.
    
    Returns: (risk_label, risk_color)
    """
    if anomaly_rate < 5:
        return "LOW", COLOR_NORMAL
    elif anomaly_rate < 15:
        return "MEDIUM", COLOR_WARNING
    else:
        return "HIGH", COLOR_ANOMALY

def apply_theme():
    """Apply CSS styling for dark theme with purple accents."""
    theme = THEMES["dark"]
    css = f"""
    <style>
        /* Global dark theme base */
        .stApp {{
            background-color: {theme['bg_primary']};
            color: {theme['text_primary']};
        }}
        
        [data-testid="stSidebar"] {{
            background-color: {theme['bg_secondary']};
            border-right: 1px solid {theme['border']};
        }}
        
        /* Typography */
        h1, h2, h3, h4, h5, h6 {{
            color: {theme['text_primary']};
            font-weight: 600;
            letter-spacing: -0.5px;
        }}
        
        h2 {{
            color: {theme['text_primary']};
            padding-bottom: 12px;
            border-bottom: 2px solid {theme['purple_dark']};
            display: inline-block;
        }}
        
        p, label {{
            color: {theme['text_secondary']};
        }}
        
        /* KPI Cards - shadcn-inspired */
        .kpi-card {{
            background: linear-gradient(135deg, {theme['bg_secondary']} 0%, {theme['bg_tertiary']} 100%);
            border: 1px solid {theme['border']};
            border-radius: 12px;
            padding: 20px;
            margin: 12px 0;
            transition: all 0.3s ease;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }}
        
        .kpi-card:hover {{
            border-color: {theme['purple_accent']};
            box-shadow: 0 0 20px rgba({int(theme['purple_accent'][1:3], 16)}, {int(theme['purple_accent'][3:5], 16)}, {int(theme['purple_accent'][5:7], 16)}, 0.3);
        }}
        
        .kpi-label {{
            font-size: 12px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 1px;
            color: {theme['text_secondary']};
            margin-bottom: 8px;
        }}
        
        .kpi-value {{
            font-size: 32px;
            font-weight: 700;
            color: {theme['text_primary']};
        }}
        
        .kpi-delta {{
            font-size: 14px;
            color: {theme['text_secondary']};
            margin-top: 8px;
        }}
        
        /* Dividers */
        hr {{
            border: none;
            height: 1px;
            background: linear-gradient(90deg, transparent, {theme['border']}, transparent);
            margin: 24px 0;
        }}
        
        /* Live indicator pulse */
        .live-indicator {{
            display: inline-block;
            background: linear-gradient(135deg, {COLOR_ANOMALY}, #DC2626);
            color: white;
            padding: 6px 14px;
            border-radius: 20px;
            font-size: 11px;
            font-weight: 700;
            letter-spacing: 1px;
            text-transform: uppercase;
            animation: pulse 1.5s infinite;
            box-shadow: 0 0 10px rgba(239, 68, 68, 0.5);
        }}
        
        @keyframes pulse {{
            0%, 100% {{ opacity: 1; }}
            50% {{ opacity: 0.6; }}
        }}
        
        /* Table styling */
        .stDataFrame {{
            background-color: {theme['bg_secondary']} !important;
            border: 1px solid {theme['border']} !important;
            border-radius: 8px !important;
            overflow: hidden;
        }}
        
        /* Anomaly alert box */
        .anomaly-alert {{
            background-color: rgba(239, 68, 68, 0.08);
            border-left: 4px solid {COLOR_ANOMALY};
            border-radius: 8px;
            padding: 16px;
            margin: 12px 0;
            transition: all 0.2s ease;
        }}
        
        .anomaly-alert:hover {{
            background-color: rgba(239, 68, 68, 0.12);
            border-left-color: {theme['purple_accent']};
        }}
        
        /* Info/Success boxes */
        .stInfo, .stSuccess {{
            background-color: {theme['bg_secondary']} !important;
            border: 1px solid {theme['border']} !important;
            border-left: 4px solid {theme['purple_accent']} !important;
            border-radius: 8px;
        }}
        
        /* Metric styling */
        [data-testid="metric-container"] {{
            background-color: {theme['bg_secondary']};
            border: 1px solid {theme['border']};
            border-radius: 12px;
            padding: 20px;
            transition: all 0.3s ease;
        }}
        
        [data-testid="metric-container"]:hover {{
            border-color: {theme['purple_accent']};
            box-shadow: 0 0 15px rgba({int(theme['purple_accent'][1:3], 16)}, {int(theme['purple_accent'][3:5], 16)}, {int(theme['purple_accent'][5:7], 16)}, 0.2);
        }}
        
        /* Streamlit input elements */
        .stRadio > label {{
            color: {theme['text_secondary']};
        }}
        
        /* Section headers with purple underline */
        .section-header {{
            color: {theme['text_primary']};
            font-size: 18px;
            font-weight: 600;
            padding-bottom: 12px;
            border-bottom: 2px solid {theme['purple_dark']};
            margin-bottom: 20px;
        }}
        
        /* Footer styling */
        .footer {{
            text-align: center;
            color: {theme['text_secondary']};
            font-size: 11px;
            border-top: 1px solid {theme['border']};
            padding-top: 20px;
            margin-top: 40px;
        }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

def add_transaction_to_stream(
    timestamp: str,
    user_id: str,
    amount: float,
    anomaly_score: float,
    is_anomaly: bool,
    explanation: str = ""
):
    """Add transaction to live stream and update metrics."""
    if not explanation and is_anomaly:
        if anomaly_score > 0.9:
            explanation = "Extreme anomaly detected"
        elif anomaly_score > 0.7:
            explanation = "High anomaly probability"
        else:
            explanation = "Moderate anomaly flagged"
    elif not explanation:
        explanation = "Normal transaction"
    
    transaction = {
        "timestamp": timestamp,
        "user_id": user_id,
        "amount": amount,
        "anomaly_score": anomaly_score,
        "is_anomaly": is_anomaly,
        "explanation": explanation,
    }
    
    st.session_state.live_transactions.append(transaction)
    st.session_state.transaction_stats["total"] += 1
    if is_anomaly:
        st.session_state.transaction_stats["anomalies"] += 1
    
    st.session_state.last_update = datetime.now()
    
    # Update time-series buffer
    current_minute = datetime.now().strftime("%H:%M")
    if current_minute not in st.session_state.anomalies_per_minute:
        st.session_state.anomalies_per_minute[current_minute] = 0
    if is_anomaly:
        st.session_state.anomalies_per_minute[current_minute] += 1

# ============================================================================
# DASHBOARD LAYOUT - GLOBAL UI SETUP
# ============================================================================

# Apply dark theme with purple accents
apply_theme()

# ============================================================================
# FUTURISTIC HEADER — "ANOMALY ENGINE"
# ============================================================================

# Create a minimal, bold header with futuristic styling
header_html = """
<style>
    .anomaly-header-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        padding: 40px 0 24px 0;
        position: relative;
    }
    
    .anomaly-header-accent-top {
        width: 100%;
        height: 2px;
        background: linear-gradient(90deg, 
            transparent, 
            #A78BFA 30%, 
            #7C3AED 50%, 
            #A78BFA 70%, 
            transparent);
        margin-bottom: 32px;
        box-shadow: 0 0 20px rgba(167, 139, 250, 0.3);
    }
    
    .anomaly-header-title {
        font-size: 42px;
        font-weight: 700;
        letter-spacing: -1px;
        color: #F1F5F9;
        text-align: center;
        margin: 0;
        padding: 0;
        background: linear-gradient(135deg, #F1F5F9 0%, #E2E8F0 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-shadow: 0 0 30px rgba(167, 139, 250, 0.2);
    }
    
    .anomaly-header-accent-bottom {
        width: 100%;
        height: 1px;
        background: linear-gradient(90deg, 
            transparent, 
            #7C3AED 30%, 
            #A78BFA 50%, 
            #7C3AED 70%, 
            transparent);
        margin-top: 24px;
        box-shadow: 0 0 15px rgba(124, 58, 237, 0.2);
    }
</style>

<div class="anomaly-header-container">
    <div class="anomaly-header-accent-top"></div>
    <h1 class="anomaly-header-title">Anomaly Engine</h1>
    <div class="anomaly-header-accent-bottom"></div>
</div>
"""

st.markdown(header_html, unsafe_allow_html=True)

st.markdown("---")

# ============================================================================
# BLOCK 1: TOP KPI CARDS - NEVER EMPTY
# ============================================================================

st.markdown('<div class="section-header">📈 Key Performance Indicators</div>', unsafe_allow_html=True)

total = st.session_state.transaction_stats["total"]
anomalies = st.session_state.transaction_stats["anomalies"]
anomaly_rate = (anomalies / total * 100) if total > 0 else 0.0
risk_level, risk_color = get_risk_level(anomaly_rate)

# Create 4 KPI columns with enhanced styling
kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4, gap="medium")

theme = THEMES["dark"]

# KPI 1: Total Events
with kpi_col1:
    total_display = f"{total:,}" if total > 0 else "—"
    total_delta = "Waiting for events" if total == 0 else "↑ Cumulative transactions"
    total_color = theme['text_secondary'] if total == 0 else theme['text_primary']
    
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">Total Events</div>
        <div class="kpi-value" style="color: {total_color};">{total_display}</div>
        <div class="kpi-delta">{total_delta}</div>
    </div>
    """, unsafe_allow_html=True)

# KPI 2: Anomalies Detected
with kpi_col2:
    anomaly_display = f"{anomalies:,}" if total > 0 else "—"
    anomaly_delta = f"📈 {anomaly_rate:.2f}%" if total > 0 else "No data yet"
    anomaly_color = COLOR_ANOMALY if anomalies > 0 else theme['text_secondary']
    
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">Anomalies Detected</div>
        <div class="kpi-value" style="color: {anomaly_color};">{anomaly_display}</div>
        <div class="kpi-delta">{anomaly_delta}</div>
    </div>
    """, unsafe_allow_html=True)

# KPI 3: Anomaly Rate
with kpi_col3:
    rate_display = f"{anomaly_rate:.2f}%" if total > 0 else "—"
    rate_color = COLOR_NORMAL if anomaly_rate < 5 else (COLOR_WARNING if anomaly_rate < 15 else COLOR_ANOMALY)
    if total == 0:
        rate_color = theme['text_secondary']
    
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">Anomaly Rate</div>
        <div class="kpi-value" style="color: {rate_color};">{rate_display}</div>
        <div class="kpi-delta">Current rate</div>
    </div>
    """, unsafe_allow_html=True)

# KPI 4: Risk Level
with kpi_col4:
    risk_indicator = "🟢" if risk_level == "LOW" else ("🟡" if risk_level == "MEDIUM" else "🔴")
    
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">Risk Level</div>
        <div class="kpi-value" style="color: {risk_color};">{risk_indicator} {risk_level}</div>
        <div class="kpi-delta">System status</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# ============================================================================
# BLOCK 5: TIME-SERIES CHARTS - ALWAYS RENDERED WITH GHOST DATA
# ============================================================================

st.markdown('<div class="section-header">📉 Real-Time Analytics</div>', unsafe_allow_html=True)

# Determine data source
has_real_data = len(st.session_state.live_transactions) > 0

if has_real_data:
    df_transactions = pd.DataFrame(list(st.session_state.live_transactions))
else:
    # Use ghost data for visual scaffolding
    df_transactions = pd.DataFrame([
        {
            "timestamp": t,
            "anomaly_score": s,
            "is_anomaly": False,
            "_is_ghost": True
        }
        for t, s in zip(ghost_time_index(30), ghost_wave_series(30, amplitude=0.08, center=0.35))
    ])

# Create 3 charts in rows
chart_row1_col1, chart_row1_col2 = st.columns(2, gap="medium")

# Chart 1: Anomaly Score Over Time
with chart_row1_col1:
    chart_label = "📊 Anomaly Score Trend" if has_real_data else "📊 Anomaly Score Trend (Baseline)"
    st.markdown(f"**{chart_label}**")
    
    df_chart = df_transactions[["timestamp", "anomaly_score", "is_anomaly"]].copy()
    df_chart["index"] = range(len(df_chart))
    
    fig = go.Figure()
    
    # Normal transactions
    normal_points = df_chart[df_chart["is_anomaly"] == False]
    normal_color = COLOR_NEUTRAL if not has_real_data else COLOR_NORMAL
    fig.add_trace(go.Scatter(
        x=normal_points["index"],
        y=normal_points["anomaly_score"],
        mode="lines+markers",
        name="Normal",
        line=dict(color=normal_color, width=2),
        marker=dict(size=5),
        opacity=0.7 if not has_real_data else 1.0,
        hovertemplate="<b>Score:</b> %{y:.4f}<extra></extra>"
    ))
    
    # Anomalous transactions
    anomaly_points = df_chart[df_chart["is_anomaly"] == True]
    if not anomaly_points.empty:
        fig.add_trace(go.Scatter(
            x=anomaly_points["index"],
            y=anomaly_points["anomaly_score"],
            mode="markers",
            name="Anomaly",
            marker=dict(color=COLOR_ANOMALY, size=10, symbol="diamond"),
            hovertemplate="<b>ANOMALY</b><br>Score: %{y:.4f}<extra></extra>"
        ))
    
    # Add threshold line
    fig.add_hline(y=0.5, line_dash="dash", line_color=COLOR_PRIMARY_PURPLE, 
                 annotation_text="Threshold", annotation_position="right")
    
    fig.update_layout(
        template="plotly_dark",
        height=350,
        hovermode="x unified",
        margin=dict(l=0, r=0, t=0, b=0),
        xaxis_title="Transaction Index",
        yaxis_title="Anomaly Score",
        showlegend=True,
        plot_bgcolor="#1E293B",
        paper_bgcolor="#0F172A",
        font=dict(color="#F1F5F9"),
        xaxis=dict(gridcolor="#334155"),
        yaxis=dict(gridcolor="#334155")
    )
    
    st.plotly_chart(fig, use_container_width=True)

# Chart 2: Anomaly Rate Rolling Window
with chart_row1_col2:
    chart_label = "📈 Rolling Anomaly Rate" if has_real_data else "📈 Rolling Anomaly Rate (Baseline)"
    st.markdown(f"**{chart_label}**")
    
    df_rolling = df_transactions.copy()
    df_rolling["is_anomaly_int"] = df_rolling["is_anomaly"].astype(int)
    window_size = min(10, len(df_rolling))
    df_rolling["rolling_rate"] = (
        df_rolling["is_anomaly_int"]
        .rolling(window=window_size, min_periods=1)
        .mean() * 100
    )
    
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        y=df_rolling["rolling_rate"],
        mode="lines",
        fill="tozeroy",
        name="Anomaly Rate %",
        line=dict(color=COLOR_WARNING, width=3),
        opacity=0.7 if not has_real_data else 1.0,
        fillcolor=f"rgba(245, 158, 11, 0.15)"
    ))
    
    fig2.update_layout(
        template="plotly_dark",
        height=350,
        hovermode="x",
        margin=dict(l=0, r=0, t=0, b=0),
        xaxis_title="Transaction Index",
        yaxis_title="Rate (%)",
        showlegend=False,
        plot_bgcolor="#1E293B",
        paper_bgcolor="#0F172A",
        font=dict(color="#F1F5F9"),
        xaxis=dict(gridcolor="#334155"),
        yaxis=dict(gridcolor="#334155")
    )
    
    st.plotly_chart(fig2, use_container_width=True)

# Chart 3: Anomalies Per Minute + Gauge
chart_row2_col1, chart_row2_col2 = st.columns([2, 1], gap="medium")

with chart_row2_col1:
    chart_label = "⏱️ Anomalies Per Minute" if has_real_data else "⏱️ Anomalies Per Minute (Baseline)"
    st.markdown(f"**{chart_label}**")
    
    if st.session_state.anomalies_per_minute:
        apm_df = pd.DataFrame(
            list(st.session_state.anomalies_per_minute.items()),
            columns=["Minute", "Count"]
        ).tail(15)  # Last 15 minutes
    else:
        # Ghost data: baseline minute counts
        apm_df = pd.DataFrame({
            "Minute": [f"{i:02d}:00" for i in range(15)],
            "Count": [1, 2, 1, 1, 2, 1, 0, 1, 2, 1, 1, 0, 2, 1, 1]
        })
    
    fig3 = go.Figure()
    bar_color = COLOR_NEUTRAL if not has_real_data else COLOR_ANOMALY
    fig3.add_trace(go.Bar(
        x=apm_df["Minute"],
        y=apm_df["Count"],
        marker=dict(color=bar_color, line=dict(color=COLOR_PRIMARY_PURPLE, width=1)),
        opacity=0.6 if not has_real_data else 1.0,
        name="Count"
    ))
    
    fig3.update_layout(
        template="plotly_dark",
        height=300,
        hovermode="x",
        margin=dict(l=0, r=0, t=0, b=0),
        xaxis_title="Minute",
        yaxis_title="Anomalies",
        showlegend=False,
        plot_bgcolor="#1E293B",
        paper_bgcolor="#0F172A",
        font=dict(color="#F1F5F9"),
        xaxis=dict(gridcolor="#334155"),
        yaxis=dict(gridcolor="#334155")
    )
    
    st.plotly_chart(fig3, use_container_width=True)

with chart_row2_col2:
    st.markdown("**🎯 Anomaly Score Gauge**")
    
    # Get current score (real or ghost baseline)
    if has_real_data and len(st.session_state.live_transactions) > 0:
        latest_tx = list(st.session_state.live_transactions)[-1]
        current_score = latest_tx.get('anomaly_score', 0.35)
        gauge_title = "Latest Score"
    else:
        current_score = 0.35
        gauge_title = "Baseline"
    
    # Create gauge chart
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=current_score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': gauge_title, 'font': {'size': 14, 'color': '#94A3B8'}},
        gauge={
            'axis': {'range': [0, 1]},
            'bar': {'color': COLOR_PRIMARY_PURPLE},
            'steps': [
                {'range': [0, 0.33], 'color': f"rgba({int(COLOR_NORMAL[1:3], 16)}, {int(COLOR_NORMAL[3:5], 16)}, {int(COLOR_NORMAL[5:7], 16)}, 0.2)"},
                {'range': [0.33, 0.66], 'color': f"rgba({int(COLOR_WARNING[1:3], 16)}, {int(COLOR_WARNING[3:5], 16)}, {int(COLOR_WARNING[5:7], 16)}, 0.2)"},
                {'range': [0.66, 1], 'color': f"rgba({int(COLOR_ANOMALY[1:3], 16)}, {int(COLOR_ANOMALY[3:5], 16)}, {int(COLOR_ANOMALY[5:7], 16)}, 0.2)"}
            ],
            'threshold': {
                'line': {'color': COLOR_PRIMARY_PURPLE, 'width': 2},
                'thickness': 0.75,
                'value': 0.5
            }
        },
        number={'valueformat': '.3f', 'font': {'size': 24, 'color': '#F1F5F9'}}
    ))
    
    fig_gauge.update_layout(
        template="plotly_dark",
        height=300,
        margin=dict(l=10, r=10, t=50, b=10),
        paper_bgcolor="#0F172A",
        font=dict(color="#F1F5F9")
    )
    
    st.plotly_chart(fig_gauge, use_container_width=True)

st.markdown("---")

# ============================================================================
# BLOCK 3: LIVE TRANSACTION FEED - ALWAYS POPULATED
# ============================================================================

st.markdown('<div class="section-header">📋 Live Event Stream</div>', unsafe_allow_html=True)

# Use real data or ghost data
if has_real_data:
    display_rows = list(st.session_state.live_transactions)
else:
    display_rows = ghost_transaction_rows(8)

if display_rows:
    df_live = pd.DataFrame(display_rows)
    
    # Display as styled dataframe with improved formatting
    display_df = df_live[[
        "timestamp", "user_id", "amount", "anomaly_score", "is_anomaly"
    ]].copy()
    
    display_df["amount"] = display_df["amount"].apply(
        lambda x: f"${x:,.2f}" if pd.notna(x) else "N/A"
    )
    display_df["anomaly_score"] = display_df["anomaly_score"].apply(
        lambda x: f"{x:.4f}" if pd.notna(x) else "N/A"
    )
    display_df["status"] = display_df["is_anomaly"].apply(
        lambda x: "🔴 ANOMALY" if x else "✓ Normal"
    )
    
    # Remove is_anomaly column, add status instead
    display_df = display_df[["timestamp", "user_id", "amount", "anomaly_score", "status"]]
    
    # Rename columns for better readability
    display_df.columns = ["Time", "User", "Amount", "Score", "Status"]
    
    def highlight_row(row):
        if "ANOMALY" in str(row["Status"]):
            return [f"background-color: rgba(239, 68, 68, 0.2); color: #FCA5A5; font-weight: bold;"] * len(row)
        elif not has_real_data:  # Ghost data - muted appearance
            return [f"background-color: transparent; color: #64748B; opacity: 0.6;"] * len(row)
        return [f"background-color: transparent; color: #F1F5F9;"] * len(row)
    
    styled_df = display_df.style.apply(highlight_row, axis=1)
    st.dataframe(styled_df, use_container_width=True, height=400)

st.markdown("---")

# ============================================================================
# BLOCK 4: CRITICAL ANOMALIES - ALWAYS VISIBLE
# ============================================================================

st.markdown('<div class="section-header">🚨 Critical Anomalies Feed</div>', unsafe_allow_html=True)

# Filter real anomalies
real_anomalies = [
    tx for tx in st.session_state.live_transactions 
    if tx.get("is_anomaly") == True
]

if real_anomalies:
    df_anomalies = pd.DataFrame(real_anomalies)
    df_anomalies = df_anomalies.iloc[::-1].reset_index(drop=True)
    
    for idx, row in df_anomalies.iterrows():
        # Score severity
        if row['anomaly_score'] > 0.9:
            severity = "🔴 CRITICAL"
            severity_color = COLOR_ANOMALY
        elif row['anomaly_score'] > 0.7:
            severity = "🟠 HIGH"
            severity_color = COLOR_WARNING
        else:
            severity = "🟡 MEDIUM"
            severity_color = COLOR_WARNING
        
        anomaly_html = f"""
        <div class="anomaly-alert">
            <div style='display: flex; justify-content: space-between; align-items: start;'>
                <div style='flex: 1;'>
                    <div style='font-weight: 700; color: {severity_color}; font-size: 14px; margin-bottom: 8px;'>
                        {severity} • Score: {row['anomaly_score']:.4f}
                    </div>
                    <div style='display: flex; gap: 16px; font-size: 12px; color: #94A3B8;'>
                        <span><strong>User:</strong> {row['user_id']}</span>
                        <span><strong>Amount:</strong> ${row['amount']:.2f}</span>
                    </div>
                    <div style='margin-top: 8px; font-size: 12px; color: {THEMES["dark"]['text_secondary']}; font-style: italic;'>
                        💡 {row['explanation']}
                    </div>
                </div>
                <div style='text-align: right; font-size: 11px; color: {THEMES["dark"]['text_secondary']}; white-space: nowrap;'>
                    {row['timestamp']}
                </div>
            </div>
        </div>
        """
        st.markdown(anomaly_html, unsafe_allow_html=True)
else:
    # Show placeholder when no anomalies
    st.markdown(f"""
    <div style='text-align: center; padding: 40px; color: #64748B;'>
        <div style='font-size: 32px; margin-bottom: 12px;'>✓</div>
        <p style='font-size: 14px; font-weight: 600;'>No Critical Anomalies</p>
        <p style='font-size: 12px; color: #475569; margin-top: 8px;'>System operating normally</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# ============================================================================
# BLOCK 6: FOOTER & SYSTEM STATUS - REDESIGNED
# ============================================================================

st.markdown("---")

footer_col1, footer_col2, footer_col3, footer_col4 = st.columns(4)

with footer_col1:
    st.caption("🔗 Backend: http://127.0.0.1:8000")

with footer_col2:
    st.caption("⚡ Refresh: 1 second")

with footer_col3:
    st.caption("📦 Buffer: Last 50 events")

with footer_col4:
    st.caption("🔄 Status: Active")

# Footer context
st.markdown("""
<div class="footer">
    <p><strong>Simulation Mode</strong> — Data is simulated but behavior is realistic. This dashboard demonstrates real-time anomaly detection capabilities.</p>
</div>
""", unsafe_allow_html=True)

# Auto-refresh every 1 second
time.sleep(1)
st.rerun()
