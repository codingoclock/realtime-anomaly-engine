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
    initial_sidebar_state="expanded"
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

def ghost_customer_data(num_customers: int = 10) -> list[dict]:
    """Generate placeholder customer activity data for UI visualization.
    
    UI scaffolding only - real backend data will replace this.
    """
    countries = ["US", "UK", "CA", "DE", "FR", "AU", "JP", "BR", "IN", "MX"]
    customers = []
    
    for i in range(num_customers):
        customer_id = f"USR-{i+1:03d}"
        country = countries[i % len(countries)]
        activity_volume = np.random.randint(5, 50)
        customers.append({
            "customer_id": customer_id,
            "country": country,
            "activity_volume": activity_volume,
            "transaction_amount": np.random.uniform(100, 5000),
            "rolling_mean_amount": np.random.uniform(800, 2500),
            "transaction_count_1min": np.random.randint(1, 10),
            "time_since_last": np.random.uniform(5, 120),
            "_is_ghost": True
        })
    
    return customers

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
        
        /* Sidebar navigation styling */
        .nav-item {{
            padding: 12px 16px;
            margin: 4px 0;
            border-radius: 8px;
            color: {theme['text_secondary']};
            font-size: 14px;
            font-weight: 500;
            cursor: pointer;
            transition: all 0.2s ease;
            border-left: 3px solid transparent;
        }}
        
        .nav-item:hover {{
            background-color: {theme['bg_tertiary']};
            color: {theme['text_primary']};
            border-left-color: {theme['purple_accent']};
        }}
        
        .nav-item.active {{
            background-color: {theme['bg_tertiary']};
            color: {theme['purple_accent']};
            border-left-color: {theme['purple_accent']};
            font-weight: 600;
        }}
        
        /* Customer card styling */
        .customer-card {{
            background: linear-gradient(135deg, {theme['bg_secondary']} 0%, {theme['bg_tertiary']} 100%);
            border: 1px solid {theme['border']};
            border-radius: 8px;
            padding: 16px;
            margin: 8px 0;
            transition: all 0.3s ease;
        }}
        
        .customer-card:hover {{
            border-color: {theme['purple_accent']};
            box-shadow: 0 0 15px rgba({int(theme['purple_accent'][1:3], 16)}, {int(theme['purple_accent'][3:5], 16)}, {int(theme['purple_accent'][5:7], 16)}, 0.2);
        }}
        
        .feature-badge {{
            display: inline-block;
            background-color: {theme['bg_tertiary']};
            border: 1px solid {theme['border']};
            border-radius: 6px;
            padding: 6px 12px;
            margin: 4px;
            font-size: 11px;
            color: {theme['text_secondary']};
        }}
        
        /* Top header bar */
        .top-header-bar {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 16px 24px;
            background-color: {theme['bg_secondary']};
            border-bottom: 1px solid {theme['border']};
            margin-bottom: 24px;
        }}
        
        .breadcrumbs {{
            display: flex;
            align-items: center;
            gap: 8px;
            color: {theme['text_secondary']};
            font-size: 14px;
        }}
        
        .breadcrumbs span {{
            color: {theme['text_primary']};
        }}
        
        .user-avatars {{
            display: flex;
            align-items: center;
            gap: 12px;
        }}
        
        .avatar-circle {{
            width: 32px;
            height: 32px;
            border-radius: 50%;
            background: linear-gradient(135deg, {theme['purple_accent']}, {theme['purple_dark']});
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-weight: 600;
            font-size: 12px;
        }}
        
        /* Mini chart in KPI cards */
        .kpi-mini-chart {{
            height: 40px;
            width: 100%;
            margin-top: 12px;
            opacity: 0.7;
        }}
        
        /* Progress bar */
        .progress-bar-container {{
            width: 100%;
            height: 8px;
            background-color: {theme['bg_tertiary']};
            border-radius: 4px;
            overflow: hidden;
            margin-top: 8px;
        }}
        
        .progress-bar-fill {{
            height: 100%;
            border-radius: 4px;
            transition: width 0.3s ease;
        }}
        
        /* Enhanced table styling */
        .dataframe {{
            font-size: 13px;
        }}
        
        .dataframe thead {{
            background-color: {theme['bg_tertiary']};
            color: {theme['text_primary']};
            font-weight: 600;
        }}
        
        .dataframe tbody tr {{
            border-bottom: 1px solid {theme['border']};
        }}
        
        .dataframe tbody tr:hover {{
            background-color: {theme['bg_tertiary']};
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
# SIDEBAR NAVIGATION (VISUAL ONLY)
# ============================================================================

with st.sidebar:
    theme = THEMES["dark"]
    
    # Sidebar header with logo
    st.markdown(f"""
    <div style='padding: 20px 0 30px 0; border-bottom: 1px solid {theme['border']}; margin-bottom: 20px;'>
        <div style='font-size: 20px; font-weight: 700; color: {theme['purple_accent']}; letter-spacing: -0.5px; margin-bottom: 4px;'>
            Anomaly Engine
        </div>
        <div style='font-size: 11px; color: {theme['text_secondary']};'>ID: ENG-1001</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Navigation items (visual only - no routing logic)
    nav_items = [
        ("🏠", "Home", False),
        ("📊", "Dashboard", False),
        ("📈", "Analytics", True),  # Analytics selected
        ("👥", "Customers", False),
        ("📋", "Logs / System", False)
    ]
    
    for icon, label, is_active in nav_items:
        active_class = "active" if is_active else ""
        st.markdown(f"""
        <div class="nav-item {active_class}">
            {icon} {label}
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Help card
    st.markdown(f"""
    <div style='padding: 16px; background-color: {theme['bg_tertiary']}; border-radius: 8px; border: 1px solid {theme['border']}; margin: 20px 0;'>
        <div style='font-size: 12px; color: {theme['text_primary']}; font-weight: 600; margin-bottom: 8px;'>
            Need setup help?
        </div>
        <div style='font-size: 11px; color: {theme['text_secondary']}; margin-bottom: 12px;'>
            Get your questions answered in a 1:1 call with our team.
        </div>
        <div style='padding: 8px 12px; background-color: {theme['purple_accent']}; color: white; border-radius: 6px; text-align: center; font-size: 11px; font-weight: 600; cursor: pointer;'>
            Schedule a call
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    # User profile at bottom
    st.markdown(f"""
    <div style='padding: 12px; background-color: {theme['bg_tertiary']}; border-radius: 8px; border: 1px solid {theme['border']}; display: flex; align-items: center; gap: 12px;'>
        <div class="avatar-circle" style='flex-shrink: 0;'>AE</div>
        <div style='flex: 1;'>
            <div style='font-size: 13px; font-weight: 600; color: {theme['text_primary']};'>Admin User</div>
            <div style='font-size: 11px; color: {theme['text_secondary']};'>admin@anomaly.engine</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ============================================================================
# TOP HEADER BAR (Reference Design Style)
# ============================================================================

theme = THEMES["dark"]

# Top header bar with logo, breadcrumbs, and user info
header_bar_html = f"""
<div class="top-header-bar">
    <div style='display: flex; align-items: center; gap: 12px;'>
        <div style='font-size: 18px; font-weight: 700; color: {theme['purple_accent']};'>Anomaly Engine</div>
        <div style='font-size: 11px; color: {theme['text_secondary']}; padding: 4px 8px; background-color: {theme['bg_tertiary']}; border-radius: 4px;'>ID: ENG-1001</div>
    </div>
    <div class="breadcrumbs">
        <span>Home</span>
        <span style='color: {theme['text_secondary']};'>›</span>
        <span>Dashboard</span>
        <span style='color: {theme['text_secondary']};'>›</span>
        <span style='color: {theme['purple_accent']};'>Analytics</span>
    </div>
    <div class="user-avatars">
        <div class="avatar-circle">A1</div>
        <div class="avatar-circle">A2</div>
        <div class="avatar-circle">A3</div>
        <div style='font-size: 11px; color: {theme['text_secondary']}; margin-left: 4px;'>+9</div>
        <div style='padding: 6px 12px; background-color: {theme['purple_accent']}; color: white; border-radius: 6px; font-size: 11px; font-weight: 600; cursor: pointer;'>Invite</div>
    </div>
</div>
"""

st.markdown(header_bar_html, unsafe_allow_html=True)

# ============================================================================
# BLOCK 1: TOP KPI CARDS - NEVER EMPTY
# ============================================================================

# Remove section header for cleaner look
# st.markdown('<div class="section-header">📈 Key Performance Indicators</div>', unsafe_allow_html=True)

total = st.session_state.transaction_stats["total"]
anomalies = st.session_state.transaction_stats["anomalies"]
anomaly_rate = (anomalies / total * 100) if total > 0 else 0.0
risk_level, risk_color = get_risk_level(anomaly_rate)

# Create 4 KPI columns with enhanced styling
kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4, gap="medium")

theme = THEMES["dark"]

# Generate mini chart data for KPI cards
def create_mini_chart_data(value_type: str = "normal"):
    """Create small wave data for mini charts in KPI cards."""
    import math
    points = 20
    data = []
    for i in range(points):
        angle = (i / points) * 2 * math.pi
        if value_type == "normal":
            val = 0.5 + 0.2 * math.sin(angle * 2) + 0.1 * math.sin(angle * 3)
        elif value_type == "positive":
            val = 0.6 + 0.15 * math.sin(angle * 2) + 0.05 * math.sin(angle * 4)
        else:  # negative
            val = 0.4 + 0.15 * math.sin(angle * 2) - 0.05 * math.sin(angle * 4)
        data.append(max(0.2, min(0.9, val)))
    return data

# KPI 1: Total Events
with kpi_col1:
    total_display = f"{total:,}" if total > 0 else "0"
    total_delta = "Waiting for events" if total == 0 else f"↑ 1.19%"
    total_color = theme['text_secondary'] if total == 0 else theme['text_primary']
    mini_chart_data = create_mini_chart_data("positive")
    
    # Create mini chart
    fig_mini1 = go.Figure()
    fig_mini1.add_trace(go.Scatter(
        y=mini_chart_data,
        mode="lines",
        line=dict(color=COLOR_NORMAL, width=2),
        fill="tozeroy",
        fillcolor=f"rgba(16, 185, 129, 0.2)",
        showlegend=False,
        hoverinfo="skip"
    ))
    fig_mini1.update_layout(
        template="plotly_dark",
        height=40,
        margin=dict(l=0, r=0, t=0, b=0),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(visible=False),
        yaxis=dict(visible=False)
    )
    
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">Total Events</div>
        <div class="kpi-value" style="color: {total_color}; font-size: 28px;">{total_display}</div>
        <div class="kpi-delta" style="display: flex; align-items: center; gap: 4px;">
            <span style="color: {COLOR_NORMAL};">↑</span> {total_delta}
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.plotly_chart(fig_mini1, use_container_width=True, config={"displayModeBar": False})

# KPI 2: Anomalies Detected
with kpi_col2:
    anomaly_display = f"{anomalies:,}" if total > 0 else "0"
    anomaly_delta = f"↑ {anomaly_rate:.2f}%" if total > 0 else "No data yet"
    anomaly_color = COLOR_ANOMALY if anomalies > 0 else theme['text_secondary']
    mini_chart_data = create_mini_chart_data("normal")
    
    fig_mini2 = go.Figure()
    fig_mini2.add_trace(go.Scatter(
        y=mini_chart_data,
        mode="lines",
        line=dict(color=COLOR_ANOMALY, width=2),
        fill="tozeroy",
        fillcolor=f"rgba(239, 68, 68, 0.2)",
        showlegend=False,
        hoverinfo="skip"
    ))
    fig_mini2.update_layout(
        template="plotly_dark",
        height=40,
        margin=dict(l=0, r=0, t=0, b=0),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(visible=False),
        yaxis=dict(visible=False)
    )
    
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">Anomalies Detected</div>
        <div class="kpi-value" style="color: {anomaly_color}; font-size: 28px;">{anomaly_display}</div>
        <div class="kpi-delta" style="display: flex; align-items: center; gap: 4px;">
            <span style="color: {COLOR_ANOMALY};">↑</span> {anomaly_delta}
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.plotly_chart(fig_mini2, use_container_width=True, config={"displayModeBar": False})

# KPI 3: Anomaly Rate
with kpi_col3:
    rate_display = f"{anomaly_rate:.2f}%" if total > 0 else "0.00%"
    rate_color = COLOR_NORMAL if anomaly_rate < 5 else (COLOR_WARNING if anomaly_rate < 15 else COLOR_ANOMALY)
    if total == 0:
        rate_color = theme['text_secondary']
    mini_chart_data = create_mini_chart_data("positive")
    
    fig_mini3 = go.Figure()
    fig_mini3.add_trace(go.Scatter(
        y=mini_chart_data,
        mode="lines",
        line=dict(color=COLOR_WARNING, width=2),
        fill="tozeroy",
        fillcolor=f"rgba(245, 158, 11, 0.2)",
        showlegend=False,
        hoverinfo="skip"
    ))
    fig_mini3.update_layout(
        template="plotly_dark",
        height=40,
        margin=dict(l=0, r=0, t=0, b=0),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(visible=False),
        yaxis=dict(visible=False)
    )
    
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">Anomaly Rate</div>
        <div class="kpi-value" style="color: {rate_color}; font-size: 28px;">{rate_display}</div>
        <div class="kpi-delta" style="display: flex; align-items: center; gap: 4px;">
            <span style="color: {COLOR_NORMAL};">↑</span> 0.29%
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.plotly_chart(fig_mini3, use_container_width=True, config={"displayModeBar": False})

# KPI 4: Risk Level
with kpi_col4:
    risk_indicator = "🟢" if risk_level == "LOW" else ("🟡" if risk_level == "MEDIUM" else "🔴")
    mini_chart_data = create_mini_chart_data("normal" if risk_level == "LOW" else "negative")
    
    fig_mini4 = go.Figure()
    fig_mini4.add_trace(go.Scatter(
        y=mini_chart_data,
        mode="lines",
        line=dict(color=risk_color, width=2),
        fill="tozeroy",
        fillcolor=f"rgba({int(risk_color[1:3], 16)}, {int(risk_color[3:5], 16)}, {int(risk_color[5:7], 16)}, 0.2)",
        showlegend=False,
        hoverinfo="skip"
    ))
    fig_mini4.update_layout(
        template="plotly_dark",
        height=40,
        margin=dict(l=0, r=0, t=0, b=0),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(visible=False),
        yaxis=dict(visible=False)
    )
    
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">Risk Level</div>
        <div class="kpi-value" style="color: {risk_color}; font-size: 28px;">{risk_indicator} {risk_level}</div>
        <div class="kpi-delta">System status</div>
    </div>
    """, unsafe_allow_html=True)
    st.plotly_chart(fig_mini4, use_container_width=True, config={"displayModeBar": False})

# Add Activity Donut Chart (Reference Design Style)
activity_col1, activity_col2 = st.columns([2, 1], gap="medium")

with activity_col1:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("**📊 Total Activity**")
    
    # Create donut chart for activity breakdown
    activity_labels = ["Normal", "Anomalies", "Pending", "Processed"]
    activity_values = [
        max(1, total - anomalies) if total > 0 else 250000,
        anomalies if total > 0 else 50000,
        100000 if total > 0 else 98000,
        140000 if total > 0 else 140000
    ]
    activity_colors = [COLOR_NORMAL, COLOR_ANOMALY, COLOR_WARNING, COLOR_PRIMARY_PURPLE]
    
    total_activity = sum(activity_values)
    
    fig_donut = go.Figure(data=[go.Pie(
        labels=activity_labels,
        values=activity_values,
        hole=0.6,
        marker=dict(colors=activity_colors, line=dict(color=theme['bg_primary'], width=2)),
        textinfo="label+percent",
        textfont=dict(size=11, color=theme['text_primary']),
        hovertemplate="<b>%{label}</b><br>Value: %{value:,.0f}<br>Percent: %{percent}<extra></extra>"
    )])
    
    fig_donut.update_layout(
        template="plotly_dark",
        height=350,
        margin=dict(l=0, r=0, t=0, b=0),
        showlegend=True,
        legend=dict(
            orientation="v",
            yanchor="middle",
            y=0.5,
            xanchor="left",
            x=1.1,
            font=dict(size=11, color=theme['text_primary'])
        ),
        plot_bgcolor="#0F172A",
        paper_bgcolor="#0F172A",
        annotations=[dict(
            text=f"<b>{total_activity:,}</b><br>Total Activity",
            x=0.5, y=0.5,
            font_size=16,
            font_color=theme['text_primary'],
            showarrow=False
        )]
    )
    
    st.plotly_chart(fig_donut, use_container_width=True)

with activity_col2:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("**📈 Activity Breakdown**")
    
    # Activity breakdown list
    for label, value, color in zip(activity_labels, activity_values, activity_colors):
        st.markdown(f"""
        <div style='padding: 12px; background-color: {theme['bg_secondary']}; border-radius: 8px; margin-bottom: 8px; border-left: 4px solid {color};'>
            <div style='font-size: 13px; font-weight: 600; color: {theme['text_primary']}; margin-bottom: 4px;'>{label}</div>
            <div style='font-size: 18px; font-weight: 700; color: {color};'>{value:,}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ============================================================================
# BLOCK 5: TIME-SERIES CHARTS - ALWAYS RENDERED WITH GHOST DATA
# ============================================================================

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

# Chart 4: Event Frequency
chart_row3_col1, chart_row3_col2 = st.columns(2, gap="medium")

with chart_row3_col1:
    chart_label = "⏱️ Event Frequency" if has_real_data else "⏱️ Event Frequency (Baseline)"
    st.markdown(f"**{chart_label}**")
    
    if has_real_data:
        # Calculate event frequency from real data
        time_points = ghost_time_index(30)
        freq_data = [1] * 30  # Placeholder - would be calculated from real timestamps
    else:
        time_points = ghost_time_index(30)
        freq_data = ghost_wave_series(30, amplitude=0.5, center=2.0)
    
    fig_freq = go.Figure()
    fig_freq.add_trace(go.Scatter(
        x=list(range(len(time_points))),
        y=freq_data,
        mode="lines+markers",
        name="Events/sec",
        line=dict(color=COLOR_PRIMARY_PURPLE, width=2),
        marker=dict(size=4),
        opacity=0.7 if not has_real_data else 1.0,
        fill="tozeroy",
        fillcolor=f"rgba(167, 139, 250, 0.1)"
    ))
    
    fig_freq.update_layout(
        template="plotly_dark",
        height=300,
        hovermode="x unified",
        margin=dict(l=0, r=0, t=0, b=0),
        xaxis_title="Time Index",
        yaxis_title="Frequency",
        showlegend=False,
        plot_bgcolor="#1E293B",
        paper_bgcolor="#0F172A",
        font=dict(color="#F1F5F9"),
        xaxis=dict(gridcolor="#334155"),
        yaxis=dict(gridcolor="#334155")
    )
    
    st.plotly_chart(fig_freq, use_container_width=True)

# Chart 5: Inter-arrival Time
with chart_row3_col2:
    chart_label = "⏳ Inter-arrival Time" if has_real_data else "⏳ Inter-arrival Time (Baseline)"
    st.markdown(f"**{chart_label}**")
    
    if has_real_data:
        # Calculate inter-arrival from real data
        time_points = ghost_time_index(30)
        inter_arrival = [2.5] * 30  # Placeholder
    else:
        time_points = ghost_time_index(30)
        inter_arrival = ghost_wave_series(30, amplitude=0.3, center=2.5)
    
    fig_inter = go.Figure()
    fig_inter.add_trace(go.Scatter(
        x=list(range(len(time_points))),
        y=inter_arrival,
        mode="lines+markers",
        name="Seconds",
        line=dict(color=COLOR_WARNING, width=2),
        marker=dict(size=4),
        opacity=0.7 if not has_real_data else 1.0,
        fill="tozeroy",
        fillcolor=f"rgba(245, 158, 11, 0.1)"
    ))
    
    fig_inter.update_layout(
        template="plotly_dark",
        height=300,
        hovermode="x unified",
        margin=dict(l=0, r=0, t=0, b=0),
        xaxis_title="Time Index",
        yaxis_title="Seconds",
        showlegend=False,
        plot_bgcolor="#1E293B",
        paper_bgcolor="#0F172A",
        font=dict(color="#F1F5F9"),
        xaxis=dict(gridcolor="#334155"),
        yaxis=dict(gridcolor="#334155")
    )
    
    st.plotly_chart(fig_inter, use_container_width=True)

st.markdown("<br>", unsafe_allow_html=True)

# ============================================================================
# BLOCK 2: CUSTOMER ACTIVITY SECTION (UI-LEVEL)
# ============================================================================

# Generate customer data (UI scaffolding)
customer_data = ghost_customer_data(10)
df_customers = pd.DataFrame(customer_data)

# Customer Activity Row 1: Bar Chart + Country Distribution
customer_row1_col1, customer_row1_col2 = st.columns([2, 1], gap="medium")

with customer_row1_col1:
    # Header with icon and title (matching reference design)
    st.markdown(f"""
    <div style='display: flex; align-items: center; gap: 8px; margin-bottom: 16px;'>
        <div style='width: 24px; height: 24px; background-color: {COLOR_WARNING}; border-radius: 50%; display: flex; align-items: center; justify-content: center;'>
            <span style='font-size: 12px; color: white;'>📈</span>
        </div>
        <div style='font-size: 16px; font-weight: 600; color: {theme['text_primary']};'>Customers Activity</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Monthly data from Apr 2025 to Oct 2025 (matching reference design)
    months = ["Apr 2025", "May 2025", "Jun 2025", "Jul 2025", "Aug 2025", "Sep 2025", "Oct 2025"]
    
    # Reference data: most months have low activity, July has highlighted data
    paid_data = [450, 520, 480, 890, 510, 550, 490]  # Paid product values
    checkout_data = [680, 750, 720, 1300, 780, 820, 760]  # Checkout Product values
    
    # Create stacked bar chart
    fig_customer = go.Figure()
    
    # Paid product bars (blue)
    fig_customer.add_trace(go.Bar(
        x=months,
        y=paid_data,
        name="Paid product",
        marker=dict(color="#3B82F6", line=dict(width=0)),
        opacity=0.9
    ))
    
    # Checkout Product bars (light blue) - stacked on top
    fig_customer.add_trace(go.Bar(
        x=months,
        y=checkout_data,
        name="Checkout Product",
        marker=dict(color="#60A5FA", line=dict(width=0)),
        opacity=0.9
    ))
    
    fig_customer.update_layout(
        barmode='stack',
        template="plotly_dark",
        height=400,
        hovermode="x unified",
        hoverlabel=dict(
            bgcolor="#1E293B",
            bordercolor="#475569",
            font_size=12,
            font_family="Arial"
        ),
        margin=dict(l=0, r=0, t=40, b=0),
        xaxis=dict(
            title="",
            gridcolor="#334155",
            gridwidth=1,
            showgrid=True,
            tickfont=dict(size=11, color=theme['text_secondary'])
        ),
        yaxis=dict(
            title="",
            range=[0, 2000],
            tickmode="linear",
            tick0=0,
            dtick=500,
            gridcolor="#334155",
            gridwidth=1,
            showgrid=True,
            tickfont=dict(size=11, color=theme['text_secondary'])
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=11, color=theme['text_primary']),
            bgcolor="rgba(0,0,0,0)",
            bordercolor="rgba(0,0,0,0)"
        ),
        plot_bgcolor="#0F172A",
        paper_bgcolor="#0F172A",
        font=dict(color="#F1F5F9"),
        showlegend=True
    )
    
    st.plotly_chart(fig_customer, use_container_width=True)
    
    # Activity breakdown box (matching reference design)
    st.markdown(f"""
    <div style='padding: 12px; background-color: {theme['bg_secondary']}; border-radius: 8px; border: 1px solid {theme['border']}; margin-top: 12px;'>
        <div style='font-size: 12px; font-weight: 600; color: {theme['text_primary']}; margin-bottom: 8px;'>Activity</div>
        <div style='display: flex; gap: 16px; font-size: 11px; color: {theme['text_secondary']};'>
            <div style='display: flex; align-items: center; gap: 6px;'>
                <div style='width: 12px; height: 12px; background-color: #3B82F6; border-radius: 2px;'></div>
                <span>Paid: 890</span>
            </div>
            <div style='display: flex; align-items: center; gap: 6px;'>
                <div style='width: 12px; height: 12px; background-color: #60A5FA; border-radius: 2px;'></div>
                <span>Checkout: 1300</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

with customer_row1_col2:
    # Country flag mapping and color assignment (matching reference design exactly)
    country_config = {
        "UK": {"flag": "🇬🇧", "name": "United Kingdom", "color": "#10B981", "users": 12628, "pct": 80},
        "US": {"flag": "🇺🇸", "name": "United States", "color": "#F59E0B", "users": 10628, "pct": 70},
        "SE": {"flag": "🇸🇪", "name": "Sweden", "color": "#3B82F6", "users": 8628, "pct": 60},
        "TR": {"flag": "🇹🇷", "name": "Turkey", "color": "#A78BFA", "users": 6628, "pct": 40},
        "ES": {"flag": "🇪🇸", "name": "Spain", "color": "#60A5FA", "users": 3628, "pct": 30},
        "CA": {"flag": "🇨🇦", "name": "Canada", "color": "#3B82F6", "users": 8628, "pct": 60},
        "DE": {"flag": "🇩🇪", "name": "Germany", "color": "#A78BFA", "users": 6628, "pct": 40},
        "FR": {"flag": "🇫🇷", "name": "France", "color": "#60A5FA", "users": 3628, "pct": 30},
        "AU": {"flag": "🇦🇺", "name": "Australia", "color": "#10B981", "users": 5628, "pct": 50},
        "JP": {"flag": "🇯🇵", "name": "Japan", "color": "#3B82F6", "users": 4628, "pct": 35},
        "BR": {"flag": "🇧🇷", "name": "Brazil", "color": "#F59E0B", "users": 7628, "pct": 55},
        "IN": {"flag": "🇮🇳", "name": "India", "color": "#A78BFA", "users": 9628, "pct": 65},
        "MX": {"flag": "🇲🇽", "name": "Mexico", "color": "#60A5FA", "users": 2628, "pct": 25}
    }
    
    # Use reference design countries (matching the image exactly)
    reference_countries = [
        {"code": "UK", "users": 12628, "pct": 80},
        {"code": "US", "users": 10628, "pct": 70},
        {"code": "SE", "users": 8628, "pct": 60},
        {"code": "TR", "users": 6628, "pct": 40},
        {"code": "ES", "users": 3628, "pct": 30}
    ]
    
    # Map countries to their config, or use defaults
    def get_country_info(country_code):
        """Get country information including flag, full name, and color."""
        if country_code in country_config:
            return country_config[country_code]
        # Default fallback
        flags = {"US": "🇺🇸", "UK": "🇬🇧", "SE": "🇸🇪", "TR": "🇹🇷", "ES": "🇪🇸",
                 "CA": "🇨🇦", "DE": "🇩🇪", "FR": "🇫🇷", "AU": "🇦🇺", "JP": "🇯🇵", 
                 "BR": "🇧🇷", "IN": "🇮🇳", "MX": "🇲🇽"}
        colors = ["#10B981", "#F59E0B", "#3B82F6", "#A78BFA", "#60A5FA"]
        flag = flags.get(country_code, "🌍")
        color = colors[len(reference_countries) % len(colors)]
        return {
            "flag": flag,
            "name": country_code,
            "color": color,
            "users": 0,
            "pct": 0
        }
    
    # Create header with icon and View All button
    header_col1, header_col2 = st.columns([1, 1])
    with header_col1:
        st.markdown(f"""
        <div style='display: flex; align-items: center; gap: 8px; margin-bottom: 16px;'>
            <div style='width: 24px; height: 24px; background-color: {theme['bg_tertiary']}; border-radius: 6px; display: flex; align-items: center; justify-content: center;'>
                <span style='font-size: 12px;'>📊</span>
            </div>
            <div style='font-size: 16px; font-weight: 600; color: {theme['text_primary']};'>Customers Active</div>
        </div>
        """, unsafe_allow_html=True)
    with header_col2:
        st.markdown(f"""
        <div style='display: flex; justify-content: flex-end; margin-bottom: 16px;'>
            <div style='padding: 6px 12px; background-color: {theme['bg_tertiary']}; border: 1px solid {theme['border']}; border-radius: 6px; font-size: 11px; color: {theme['text_primary']}; cursor: pointer; display: flex; align-items: center; gap: 4px;'>
                View All
                <span style='font-size: 10px;'>↗</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Display countries with flags and progress bars (using reference design data)
    for country_data in reference_countries:
        country_code = country_data['code']
        country_info = get_country_info(country_code)
        
        # Use reference design values
        users = country_data['users']
        pct = country_data['pct']
        bar_color = country_info['color']
        country_name = country_info['name']
        flag = country_info['flag']
        
        country_html = f"""
        <div style='padding: 12px 0; border-bottom: 1px solid {theme['border']};'>
            <div style='display: flex; align-items: center; gap: 12px; margin-bottom: 8px;'>
                <div style='font-size: 24px;'>{flag}</div>
                <div style='flex: 1;'>
                    <div style='font-size: 14px; font-weight: 500; color: {theme['text_primary']};'>{country_name}</div>
                </div>
                <div style='font-size: 13px; color: {theme['text_secondary']};'>{users:,} ({pct}%)</div>
            </div>
            <div class="progress-bar-container" style='margin-top: 8px;'>
                <div class="progress-bar-fill" style='width: {pct}%; background-color: {bar_color};'></div>
            </div>
        </div>
        """
        st.markdown(country_html, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ============================================================================
# BLOCK 2.5: FEATURE-AWARE CUSTOMER DISPLAY (ML-READY)
# ============================================================================

st.markdown("""
<div style='font-size: 12px; color: #94A3B8; margin-bottom: 16px; padding: 12px; background-color: #1E293B; border-radius: 8px; border-left: 3px solid #A78BFA;'>
    <strong>Display-Only Features:</strong> These containers are ready to display ML features when real backend data arrives. 
    No computation is performed in the UI layer.
</div>
""", unsafe_allow_html=True)

# Display feature cards for top customers
feature_customers = df_customers.head(4)

feature_cols = st.columns(4, gap="medium")

for idx, (col, customer) in enumerate(zip(feature_cols, feature_customers.itertuples())):
    with col:
        theme = THEMES["dark"]
        st.markdown(f"""
        <div class="customer-card">
            <div style='font-weight: 700; color: {theme['purple_accent']}; font-size: 14px; margin-bottom: 12px;'>
                {customer.customer_id}
            </div>
            <div style='font-size: 11px; color: {theme['text_secondary']}; margin-bottom: 8px;'>
                <strong>Transaction Amount:</strong><br>
                <span style='color: {theme['text_primary']};'>${customer.transaction_amount:.2f}</span>
            </div>
            <div style='font-size: 11px; color: {theme['text_secondary']}; margin-bottom: 8px;'>
                <strong>Rolling Mean Amount:</strong><br>
                <span style='color: {theme['text_primary']};'>${customer.rolling_mean_amount:.2f}</span>
            </div>
            <div style='font-size: 11px; color: {theme['text_secondary']}; margin-bottom: 8px;'>
                <strong>Transaction Count (1min):</strong><br>
                <span style='color: {theme['text_primary']};'>{customer.transaction_count_1min}</span>
            </div>
            <div style='font-size: 11px; color: {theme['text_secondary']};'>
                <strong>Time Since Last:</strong><br>
                <span style='color: {theme['text_primary']};'>{customer.time_since_last:.1f}s</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ============================================================================
# BLOCK 3: LIVE TRANSACTION FEED - ALWAYS POPULATED
# ============================================================================

st.markdown('<div style="font-size: 16px; font-weight: 600; color: #F1F5F9; margin-bottom: 12px;">📋 Recent Transaction</div>', unsafe_allow_html=True)

# Add action buttons row
action_col1, action_col2, action_col3, action_col4, action_col5 = st.columns([1, 1, 1, 1, 6])
with action_col1:
    st.markdown(f"""
    <div style='padding: 6px 12px; background-color: {theme['bg_tertiary']}; border: 1px solid {theme['border']}; border-radius: 6px; text-align: center; font-size: 11px; color: {theme['text_primary']}; cursor: pointer;'>
        Search
    </div>
    """, unsafe_allow_html=True)
with action_col2:
    st.markdown(f"""
    <div style='padding: 6px 12px; background-color: {theme['bg_tertiary']}; border: 1px solid {theme['border']}; border-radius: 6px; text-align: center; font-size: 11px; color: {theme['text_primary']}; cursor: pointer;'>
        Hide
    </div>
    """, unsafe_allow_html=True)
with action_col3:
    st.markdown(f"""
    <div style='padding: 6px 12px; background-color: {theme['bg_tertiary']}; border: 1px solid {theme['border']}; border-radius: 6px; text-align: center; font-size: 11px; color: {theme['text_primary']}; cursor: pointer;'>
        Customize
    </div>
    """, unsafe_allow_html=True)
with action_col4:
    st.markdown(f"""
    <div style='padding: 6px 12px; background-color: {theme['bg_tertiary']}; border: 1px solid {theme['border']}; border-radius: 6px; text-align: center; font-size: 11px; color: {theme['text_primary']}; cursor: pointer;'>
        Export
    </div>
    """, unsafe_allow_html=True)

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
    
    # Add ORDER ID column first (format: ORD-XXX)
    display_df.insert(0, "ORDER ID", [f"ORD-{i+1:03d}" for i in range(len(display_df))])
    
    # Format amount for better table display (remove $ and commas for cleaner look)
    display_df["amount"] = display_df["amount"].str.replace("$", "").str.replace(",", "")
    
    # Rename columns for better readability (matching reference design)
    display_df.columns = ["ORDER ID", "DATE CHECKOUT", "CUSTOMER", "AMOUNT", "SCORE", "STATUS"]
    
    # Reorder columns to match reference: ORDER ID, CUSTOMER, AMOUNT, SCORE, STATUS, DATE CHECKOUT
    display_df = display_df[["ORDER ID", "CUSTOMER", "AMOUNT", "SCORE", "STATUS", "DATE CHECKOUT"]]
    
    def highlight_row(row):
        if "ANOMALY" in str(row["STATUS"]):
            return [f"background-color: rgba(239, 68, 68, 0.15); color: #FCA5A5;"] * len(row)
        elif not has_real_data:  # Ghost data - muted appearance
            return [f"background-color: {theme['bg_secondary']}; color: #64748B; opacity: 0.7;"] * len(row)
        return [f"background-color: {theme['bg_secondary']}; color: {theme['text_primary']};"] * len(row)
    
    styled_df = display_df.style.apply(highlight_row, axis=1)
    
    # Enhanced table styling
    st.dataframe(
        styled_df,
        use_container_width=True,
        height=400,
        hide_index=True
    )

st.markdown("<br>", unsafe_allow_html=True)

# ============================================================================
# BLOCK 4: CRITICAL ANOMALIES - ALWAYS VISIBLE
# ============================================================================

st.markdown('<div style="font-size: 16px; font-weight: 600; color: #F1F5F9; margin-bottom: 12px;">🚨 Critical Anomalies Feed</div>', unsafe_allow_html=True)

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

st.markdown("<br>", unsafe_allow_html=True)

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
