"""Simple consumer script that generates transactions and computes features.

- Initializes `EventGenerator` and `FeaturePipeline` once
- Loads `IsolationForestPredictor` at startup (if available)
- For each generated event: compute features, predict anomaly_score & is_anomaly, and print them
- No database persistence
- Exits after 50 events (safety limit)

Usage:
    python consumer/consumer.py
"""
from __future__ import annotations

import argparse
import json
import time
from typing import Optional
import os
# Import with robust fallback so the script works when run directly


try:
    from producer.event_generator import EventGenerator
    from consumer.feature_pipeline import FeaturePipeline
except Exception:
    import importlib.util
    import os
    import sys

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    # load event_generator
    eg_path = os.path.join(repo_root, "producer", "event_generator.py")
    spec = importlib.util.spec_from_file_location("producer.event_generator", eg_path)
    if spec is None or spec.loader is None:
        raise ImportError("Could not load producer.event_generator module")
    eg_mod = importlib.util.module_from_spec(spec)
    # Register module so decorators like @dataclass can access module globals
    import sys as _sys
    _sys.modules[spec.name] = eg_mod
    spec.loader.exec_module(eg_mod)
    EventGenerator = eg_mod.EventGenerator

    # load feature_pipeline
    fp_path = os.path.join(repo_root, "consumer", "feature_pipeline.py")
    spec2 = importlib.util.spec_from_file_location("consumer.feature_pipeline", fp_path)
    if spec2 is None or spec2.loader is None:
        raise ImportError("Could not load consumer.feature_pipeline module")
    fp_mod = importlib.util.module_from_spec(spec2)
    _sys.modules[spec2.name] = fp_mod
    spec2.loader.exec_module(fp_mod)
    FeaturePipeline = fp_mod.FeaturePipeline


def _explain_anomaly(features: dict) -> list:
    """Return a short list of human-readable explanations derived from feature values.

    Explanations are diagnostic only and do not affect model predictions.
    Operates solely on the feature dict returned by FeaturePipeline.
    """
    exps = []
    try:
        amt = float(features.get('transaction_amount', float('nan')))
    except Exception:
        amt = float('nan')
    try:
        mean = float(features.get('rolling_mean_amount_per_user', float('nan')))
    except Exception:
        mean = float('nan')
    try:
        count = int(features.get('transaction_count_last_1_min', 0))
    except Exception:
        count = 0
    try:
        tdelta = float(features.get('time_since_last_transaction_seconds', float('nan')))
    except Exception:
        tdelta = float('nan')

    # Rapid repeat transactions
    if not (tdelta != tdelta):  # check for NaN
        if tdelta <= 5.0:
            exps.append('very small time_since_last_transaction_seconds')
        elif tdelta <= 30.0:
            exps.append('small time_since_last_transaction_seconds')
        elif tdelta > 3600.0:
            exps.append('very long time_since_last_transaction_seconds')

    # Bursty activity
    if count >= 5:
        exps.append('unusually high transaction_count_last_1_min')
    elif count >= 3:
        exps.append('elevated transaction_count_last_1_min')

    # Amount vs rolling mean
    if mean == mean and mean > 0.0 and amt == amt:
        rel = abs(amt - mean) / mean
        if rel > 3.0:
            exps.append('large deviation between transaction_amount and rolling_mean_amount_per_user')
        elif rel > 1.0:
            exps.append('notable deviation between transaction_amount and rolling_mean_amount_per_user')
    else:
        # If mean is not available or zero, check absolute amount
        if amt == amt and amt >= 50000.0:
            exps.append('very large transaction_amount')
        elif amt == amt and amt >= 5000.0:
            exps.append('large transaction_amount')

    # Fallback explanation
    if not exps:
        exps.append('anomalous pattern detected')

    # Keep explanations concise and deterministic order
    return exps


def main() -> None:
    parser = argparse.ArgumentParser(description="Local consumer: generate events and compute features")
    parser.add_argument("--interval", type=float, default=1.0, help="Seconds to sleep between events")
    parser.add_argument("--count", type=int, default=50, help="Number of events to process (max 50)")
    parser.add_argument("--seed", type=int, default=None, help="Optional RNG seed for deterministic output")
    # Default model path points to top-level `models/isolation_forest.joblib`
    default_model_path = os.path.join(os.path.dirname(__file__), "..", "models", "isolation_forest.joblib")
    parser.add_argument("--model-path", type=str, default=default_model_path, help="Path to IsolationForest .joblib model (defaults to models/isolation_forest.joblib)")
    parser.add_argument("--threshold", type=float, default= 0.45, help="Anomaly score threshold; sample is anomalous when score >= threshold")

    # Alert-related arguments
    parser.add_argument("--alert-score-threshold", type=float, default=None, help="Score threshold to trigger score-based alerts")
    parser.add_argument("--alert-score-trigger", choices=["above", "below"], default="above", help="Trigger direction for score-based alerts")
    parser.add_argument("--rate-limit", type=int, default=10, help="Number of anomalies allowed in rate window before alerting")
    parser.add_argument("--rate-window", type=int, default=60, help="Rate window size in seconds")
    parser.add_argument("--rate-scope", choices=["global", "user"], default="global", help="Scope for rate-based alerts: global or user")

    args = parser.parse_args()

    gen = EventGenerator(seed=args.seed)
    pipeline = FeaturePipeline()

    # Enforce maximum of 50 events per run (safety limit)
    if args.count <= 0 or args.count > 50:
        args.count = 50

    # Load predictor once at startup. Failure to load is fatal (fail loudly)
    from models.predict import IsolationForestPredictor

    try:
        predictor = IsolationForestPredictor(model_path=args.model_path)
        predictor_feature_order = getattr(predictor, "feature_order", None)
        print("Loaded predictor; feature_order=", predictor_feature_order)
    except Exception as exc:
        raise RuntimeError(f"Failed to load IsolationForestPredictor from {args.model_path}: {type(exc).__name__}: {exc}") from exc

    # Initialize AlertManager (optional)
    try:
        from alerts.alert_manager import AlertManager

        alert_manager = AlertManager(
            score_threshold=args.alert_score_threshold,
            score_trigger=args.alert_score_trigger,
            rate_limit=args.rate_limit,
            rate_window_seconds=args.rate_window,
            rate_scope=args.rate_scope,
        )
    except Exception as exc:
        alert_manager = None
        print("Warning: Could not initialize AlertManager; alerts disabled:", type(exc).__name__, exc)

    print(f"Starting consumer: interval={args.interval}s, count={'infinite' if args.count<=0 else args.count}")
    processed = 0

    try:
        while args.count <= 0 or processed < args.count:
            event = gen.generate_transaction()
            features = pipeline.process_event(event)

            # Perform anomaly scoring and print exact failure reason if it fails
            try:
                anomaly_score, is_anomaly = predictor.predict(features, threshold=args.threshold)
                # Validate outputs
                if anomaly_score is None:
                    raise RuntimeError("Predictor returned None for anomaly_score")
                if is_anomaly is None:
                    raise RuntimeError("Predictor returned None for is_anomaly flag")
            except KeyError as exc:  # missing features
                print("Anomaly scoring failed: Missing required feature(s):", exc.args)
                anomaly_score = None
                is_anomaly = None
            except Exception as exc:
                # Print exact exception type and message (do not suppress silently)
                print("Anomaly scoring failed:", type(exc).__name__, exc)
                anomaly_score = None
                is_anomaly = None

            # Print summary: event id, features, and anomaly results
            print(f"EVENT: id={event.get('transaction_id')} user={event.get('user_id')} amount={event.get('amount')}")
            print("FEATURES:", json.dumps(features, ensure_ascii=False))

            if anomaly_score is None:
                print("ANOMALY: scoring failed or unavailable")
            else:
                print(f"ANOMALY_SCORE: {anomaly_score:.6f} IS_ANOMALY: {is_anomaly}")

                # If model flagged anomaly, provide short human-readable explanations derived only from features
                if is_anomaly:
                    explanations = _explain_anomaly(features)
                    # Print each explanation on its own line for clarity
                    for e in explanations:
                        print("EXPLANATION:", e)

            # Pass to AlertManager if available and scoring was performed
            if alert_manager is not None and anomaly_score is not None and is_anomaly is not None:
                alert_manager.check_and_alert(event, anomaly_score=anomaly_score, is_anomaly=is_anomaly)

            processed += 1
            time.sleep(max(0.0, float(args.interval)))
    except KeyboardInterrupt:
        print("\nKeyboardInterrupt received — shutting down consumer gracefully.")
    finally:
        print(f"Consumer stopped after processing {processed} events.")


if __name__ == "__main__":
    main()
