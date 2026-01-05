"""Unit tests for alerts.AlertManager"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import io
import sys

from alerts.alert_manager import AlertManager


def _iso(dt):
    return dt.isoformat()


def capture_print(func, *args, **kwargs):
    old_out = sys.stdout
    sys.stdout = io.StringIO()
    try:
        func(*args, **kwargs)
        return sys.stdout.getvalue()
    finally:
        sys.stdout = old_out


def test_score_alert_triggers():
    am = AlertManager(score_threshold=5.0, score_trigger='above')
    event = {'timestamp': _iso(datetime.now(timezone.utc)), 'user_id': 'u1'}
    out = capture_print(am.check_and_alert, event, anomaly_score=5.1, is_anomaly=False)
    assert '[ALERT][SCORE]' in out


def test_rate_alert_global_triggers():
    am = AlertManager(rate_limit=2, rate_window_seconds=60, rate_scope='global')
    base = datetime.now(timezone.utc)
    # three anomalous events within window -> trigger on third
    e1 = {'timestamp': _iso(base)}
    e2 = {'timestamp': _iso(base + timedelta(seconds=10))}
    e3 = {'timestamp': _iso(base + timedelta(seconds=20))}

    out1 = capture_print(am.check_and_alert, e1, anomaly_score=10.0, is_anomaly=True)
    out2 = capture_print(am.check_and_alert, e2, anomaly_score=10.0, is_anomaly=True)
    out3 = capture_print(am.check_and_alert, e3, anomaly_score=10.0, is_anomaly=True)

    assert '[ALERT][RATE][GLOBAL]' in out3


def test_rate_alert_user_triggers():
    am = AlertManager(rate_limit=2, rate_window_seconds=60, rate_scope='user')
    base = datetime.now(timezone.utc)
    e1 = {'timestamp': _iso(base), 'user_id': 'bob'}
    e2 = {'timestamp': _iso(base + timedelta(seconds=10)), 'user_id': 'bob'}
    e3 = {'timestamp': _iso(base + timedelta(seconds=20)), 'user_id': 'bob'}

    capture_print(am.check_and_alert, e1, anomaly_score=10.0, is_anomaly=True)
    capture_print(am.check_and_alert, e2, anomaly_score=10.0, is_anomaly=True)
    out3 = capture_print(am.check_and_alert, e3, anomaly_score=10.0, is_anomaly=True)

    assert '[ALERT][RATE][USER]' in out3


def test_eviction_works():
    am = AlertManager(rate_limit=2, rate_window_seconds=30, rate_scope='global')
    base = datetime.now(timezone.utc)
    e1 = {'timestamp': _iso(base - timedelta(seconds=40))}
    e2 = {'timestamp': _iso(base - timedelta(seconds=20))}
    e3 = {'timestamp': _iso(base)}

    # first event is out of window when third arrives
    capture_print(am.check_and_alert, e1, anomaly_score=10.0, is_anomaly=True)
    capture_print(am.check_and_alert, e2, anomaly_score=10.0, is_anomaly=True)
    out3 = capture_print(am.check_and_alert, e3, anomaly_score=10.0, is_anomaly=True)

    # only two events in window -> count 2 <= limit 2 -> no alert
    assert '[ALERT][RATE][GLOBAL]' not in out3
