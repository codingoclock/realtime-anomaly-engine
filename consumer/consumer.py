"""Simple consumer script that generates transactions and computes features.

- Initializes `EventGenerator` once and `FeaturePipeline` once
- Generates events locally (no streaming system)
- For each event: compute features and print the raw event and features
- Rate is controlled via `time.sleep(interval)` and `--count` for finite runs
- Graceful shutdown via KeyboardInterrupt (Ctrl+C)

Usage:
    python consumer/consumer.py --interval 1.0 --count 0
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Local consumer: generate events and compute features")
    parser.add_argument("--interval", type=float, default=1.0, help="Seconds to sleep between events")
    parser.add_argument("--count", type=int, default=0, help="Number of events to process (0 = infinite)")
    parser.add_argument("--seed", type=int, default=None, help="Optional RNG seed for deterministic output")
    parser.add_argument("--model-path", type=str, default=None, help="Path to IsolationForest .joblib model (optional)")
    parser.add_argument("--threshold", type=float, default=0.0, help="Anomaly score threshold; sample is anomalous when score >= threshold")
    parser.add_argument("--fill-missing-with-zero", action="store_true", help="If set, missing model features will be filled with 0.0 when building input vector from pipeline features")

    # Alert-related arguments
    parser.add_argument("--alert-score-threshold", type=float, default=None, help="Score threshold to trigger score-based alerts")
    parser.add_argument("--alert-score-trigger", choices=["above", "below"], default="above", help="Trigger direction for score-based alerts")
    parser.add_argument("--rate-limit", type=int, default=10, help="Number of anomalies allowed in rate window before alerting")
    parser.add_argument("--rate-window", type=int, default=60, help="Rate window size in seconds")
    parser.add_argument("--rate-scope", choices=["global", "user"], default="global", help="Scope for rate-based alerts: global or user")
    parser.add_argument("--db-path", type=str, default=None, help="Optional path to sqlite DB file for persistence")

    args = parser.parse_args()

    gen = EventGenerator(seed=args.seed)
    pipeline = FeaturePipeline()

    # Load predictor once at startup. If loading fails we continue without scoring
    try:
        from models.predict import IsolationForestPredictor

        predictor = IsolationForestPredictor(model_path=args.model_path)
        predictor_feature_order = getattr(predictor, "feature_order", None)
    except Exception as exc:
        predictor = None
        predictor_feature_order = None
        print("Warning: Could not load predictor; anomaly scoring will be disabled:", type(exc).__name__, exc)

    # Initialize AlertManager
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

    # Initialize DB persistence (optional). If DB init fails we continue without persistence.
    try:
        from storage.database import init_db, insert_processed_event

        db_path = args.db_path or os.environ.get("DB_PATH")
        init_db(db_path)

        _insert_processed_event = insert_processed_event
    except Exception as exc:
        _insert_processed_event = None
        print("Warning: Persistence disabled (DB init failed):", type(exc).__name__, exc)

    print(f"Starting consumer: interval={args.interval}s, count={'infinite' if args.count<=0 else args.count}")
    processed = 0

    try:
        while args.count <= 0 or processed < args.count:
            event = gen.generate_transaction()
            features = pipeline.process_event(event)

            # Prepare and perform anomaly scoring if predictor is available
            anomaly_score = None
            is_anomaly = None

            if predictor is not None:
                if predictor_feature_order is not None:
                    input_dict = {}
                    missing = []
                    for name in predictor_feature_order:
                        if name in features:
                            input_dict[name] = features[name]
                        else:
                            missing.append(name)
                            input_dict[name] = 0.0 if args.fill_missing_with_zero else None

                    if not args.fill_missing_with_zero and any(v is None for v in input_dict.values()):
                        # Skip scoring if required features are missing and we're not filling
                        print("FEATURES:", json.dumps(features, ensure_ascii=False))
                        print("Anomaly scoring skipped: predictor expects features", predictor_feature_order)
                    else:
                        if missing and args.fill_missing_with_zero:
                            print("Note: filling missing features with 0.0 for predictor:", missing)
                        anomaly_score, is_anomaly = predictor.predict(input_dict, threshold=args.threshold)
                else:
                    try:
                        anomaly_score, is_anomaly = predictor.predict(features, threshold=args.threshold)
                    except Exception as exc:
                        print("Anomaly scoring failed:", type(exc).__name__, exc)
                        anomaly_score = None
                        is_anomaly = None

            # Print raw event, computed features, and anomaly results
            print("EVENT:", json.dumps(event, ensure_ascii=False))
            print("FEATURES:", json.dumps(features, ensure_ascii=False))
            print("ANOMALY_SCORE:", anomaly_score)
            print("IS_ANOMALY:", is_anomaly)

            # Pass to AlertManager if available and scoring was performed
            if alert_manager is not None and anomaly_score is not None and is_anomaly is not None:
                alert_manager.check_and_alert(event, anomaly_score=anomaly_score, is_anomaly=is_anomaly)

            # Persist processed event (non-fatal)
            if _insert_processed_event is not None:
                try:
                    _insert_processed_event(
                        event,
                        features,
                        anomaly_score if anomaly_score is not None else float('nan'),
                        bool(is_anomaly) if is_anomaly is not None else False,
                        db_path=db_path,
                        processed_ts=None,
                    )
                except Exception as exc:
                    print("Warning: DB insert failed (continuing):", type(exc).__name__, exc)

            processed += 1
            time.sleep(max(0.0, float(args.interval)))
    except KeyboardInterrupt:
        print("\nKeyboardInterrupt received — shutting down consumer gracefully.")
    finally:
        print(f"Consumer stopped after processing {processed} events.")


if __name__ == "__main__":
    main()
