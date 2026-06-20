"""Scoring consumer.

Reads transaction events from either:
  - Apache Kafka topic `transactions` (default)
  - In-process EventGenerator (--source generator, for offline demos / tests)

For each event:
  1. FeaturePipeline extracts 4 behavioural features
  2. IsolationForestPredictor scores the event
  3. Persists raw_event + features + anomaly result to SQLite
  4. Publishes a scored-events message to Kafka (in kafka mode)
  5. AlertManager fires console alerts when thresholds are crossed

Usage:
    python consumer/consumer.py                         # Kafka source (default)
    python consumer/consumer.py --source generator      # in-process generator (offline)
"""
from __future__ import annotations

import argparse
import json
import sys
import os
import time
from datetime import datetime, timezone
from typing import Optional

# ── Path bootstrap ────────────────────────────────────────────────────────────
_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.dirname(_here)
for _p in (_root, _here):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from producer.event_generator import EventGenerator
from feature_pipeline import FeaturePipeline   # consumer/ is in sys.path
from storage import database
import config


# ── Human-readable anomaly explanations ──────────────────────────────────────

def _explain_anomaly(features: dict) -> list[str]:
    """Return short human-readable explanations derived from feature values."""
    exps: list[str] = []
    try:
        amt = float(features.get("transaction_amount", float("nan")))
    except Exception:
        amt = float("nan")
    try:
        mean = float(features.get("rolling_mean_amount_per_user", float("nan")))
    except Exception:
        mean = float("nan")
    try:
        count = int(features.get("transaction_count_last_1_min", 0))
    except Exception:
        count = 0
    try:
        tdelta = float(features.get("time_since_last_transaction_seconds", float("nan")))
    except Exception:
        tdelta = float("nan")

    if tdelta == tdelta:  # not NaN
        if tdelta <= 5.0:
            exps.append("very small time_since_last_transaction_seconds")
        elif tdelta <= 30.0:
            exps.append("small time_since_last_transaction_seconds")
        elif tdelta > 3600.0:
            exps.append("very long time_since_last_transaction_seconds")

    if count >= 5:
        exps.append("unusually high transaction_count_last_1_min")
    elif count >= 3:
        exps.append("elevated transaction_count_last_1_min")

    if mean == mean and mean > 0.0 and amt == amt:
        rel = abs(amt - mean) / mean
        if rel > 3.0:
            exps.append("large deviation between transaction_amount and rolling_mean_amount_per_user")
        elif rel > 1.0:
            exps.append("notable deviation between transaction_amount and rolling_mean_amount_per_user")
    else:
        if amt == amt and amt >= 50000.0:
            exps.append("very large transaction_amount")
        elif amt == amt and amt >= 5000.0:
            exps.append("large transaction_amount")

    if not exps:
        exps.append("anomalous pattern detected")
    return exps


# ── Scored-events message builder ─────────────────────────────────────────────

def build_scored_message(
    event: dict,
    features: dict,
    anomaly_score: float,
    is_anomaly: bool,
    explanations: list[str],
    anomaly_row_id: Optional[int] = None,
    processed_timestamp: Optional[str] = None,
) -> dict:
    """Build the scored-events Kafka message dict.

    Schema version 1 — all numeric values are native Python types so the dict
    is directly JSON-serialisable without a custom encoder.
    """
    return {
        "schema_version": 1,
        "transaction_id": str(event.get("transaction_id", "")),
        "user_id": str(event.get("user_id", "")),
        "amount": float(event.get("amount", 0.0)),
        "timestamp": str(event.get("timestamp", "")),
        "processed_timestamp": processed_timestamp or datetime.now(timezone.utc).isoformat(),
        "anomaly_score": float(anomaly_score),
        "is_anomaly": bool(is_anomaly),
        "explanation": explanations,
        "features": {k: float(v) for k, v in features.items()},
        "event": dict(event),
        "anomaly_row_id": int(anomaly_row_id) if anomaly_row_id is not None else None,
    }


# ── Per-event processing ──────────────────────────────────────────────────────

def process_one(
    event: dict,
    pipeline: FeaturePipeline,
    predictor,
    alert_manager,
    threshold: float,
    db_path: str,
    fill_missing: bool,
    kafka_out,
    output_topic: str,
) -> None:
    """Process a single transaction event end-to-end."""
    # 1. Feature extraction
    try:
        features = pipeline.process_event(event)
    except Exception as exc:
        print(f"[WARN] Feature extraction failed: {exc}")
        if fill_missing:
            features = {
                "transaction_amount": float(event.get("amount", 0.0)),
                "rolling_mean_amount_per_user": 0.0,
                "transaction_count_last_1_min": 0,
                "time_since_last_transaction_seconds": 0.0,
            }
        else:
            return

    # 2. Anomaly scoring
    try:
        anomaly_score, is_anomaly = predictor.predict(features, threshold=threshold)
        if anomaly_score is None or is_anomaly is None:
            raise RuntimeError("Predictor returned None")
    except Exception as exc:
        print(f"[WARN] Scoring failed: {type(exc).__name__}: {exc}")
        if fill_missing:
            anomaly_score, is_anomaly = 0.0, False
        else:
            return

    # 3. Persist to SQLite (every event — not just anomalies)
    anomaly_row_id: Optional[int] = None
    processed_ts = datetime.now(timezone.utc).isoformat()
    try:
        database.init_db(db_path)
        anomaly_row_id = database.insert_processed_event(
            event, features, float(anomaly_score), bool(is_anomaly),
            db_path=db_path, processed_ts=processed_ts,
        )
    except Exception as exc:
        print(f"[WARN] DB persistence failed: {type(exc).__name__}: {exc}")

    # 4. Human-readable explanation
    explanations = _explain_anomaly(features) if is_anomaly else []

    # 5. Console output
    print(f"EVENT: id={event.get('transaction_id')} user={event.get('user_id')} amount={event.get('amount')}")
    print(f"FEATURES: {json.dumps(features, ensure_ascii=False)}")
    print(f"ANOMALY_SCORE: {anomaly_score:.6f}  IS_ANOMALY: {is_anomaly}")
    if is_anomaly and explanations:
        for e in explanations:
            print(f"EXPLANATION: {e}")

    # 6. Alert manager
    if alert_manager is not None:
        try:
            alert_manager.check_and_alert(event, anomaly_score=float(anomaly_score), is_anomaly=bool(is_anomaly))
        except Exception as exc:
            print(f"[WARN] Alert check failed: {exc}")

    # 7. Publish to scored-events topic
    if kafka_out is not None:
        try:
            msg = build_scored_message(
                event, features, float(anomaly_score), bool(is_anomaly),
                explanations, anomaly_row_id, processed_ts,
            )
            kafka_out.send(output_topic, key=event.get("user_id"), value=msg)
        except Exception as exc:
            print(f"[WARN] scored-events publish failed: {exc}")


# ── Kafka helpers ─────────────────────────────────────────────────────────────

def _build_kafka_consumer(bootstrap_servers: str, topic: str, group_id: str):
    from kafka import KafkaConsumer  # type: ignore[import]

    return KafkaConsumer(
        topic,
        bootstrap_servers=bootstrap_servers,
        group_id=group_id,
        value_deserializer=lambda b: json.loads(b.decode("utf-8")),
        key_deserializer=lambda b: b.decode("utf-8") if b else None,
        auto_offset_reset="latest",
        enable_auto_commit=True,
        consumer_timeout_ms=-1,  # block forever
    )


def _build_kafka_producer(bootstrap_servers: str):
    from kafka import KafkaProducer  # type: ignore[import]

    return KafkaProducer(
        bootstrap_servers=bootstrap_servers,
        value_serializer=lambda v: json.dumps(v, ensure_ascii=False).encode("utf-8"),
        key_serializer=lambda k: k.encode("utf-8") if k else None,
        linger_ms=5,
        acks="all",
        retries=3,
    )


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    default_model = os.path.join(os.path.dirname(__file__), "..", "models", "isolation_forest.joblib")

    parser = argparse.ArgumentParser(description="Realtime anomaly scoring consumer")

    # Source
    parser.add_argument(
        "--source", choices=["kafka", "generator"], default="kafka",
        help="Event source: 'kafka' (default) or in-process 'generator'",
    )
    parser.add_argument("--bootstrap-servers", default=config.KAFKA_BOOTSTRAP_SERVERS)
    parser.add_argument("--input-topic", default=config.TOPIC_TRANSACTIONS)
    parser.add_argument("--output-topic", default=config.TOPIC_SCORED_EVENTS)
    parser.add_argument("--group-id", default=config.KAFKA_GROUP_SCORING)

    # Generator fallback options (used when --source generator)
    parser.add_argument("--interval", type=float, default=1.0, help="Seconds between events (generator mode)")
    parser.add_argument("--count", type=int, default=50, help="Number of events (generator mode, max 50)")
    parser.add_argument("--seed", type=int, default=None)

    # Model
    parser.add_argument("--model-path", default=default_model)
    parser.add_argument("--threshold", type=float, default=0.45)

    # Persistence
    parser.add_argument("--db-path", default=config.DB_PATH, help="SQLite database path")
    parser.add_argument("--fill-missing-with-zero", action="store_true",
                        help="Default failed features/scores to 0 instead of skipping")

    # Alert flags
    parser.add_argument("--alert-score-threshold", type=float, default=None)
    parser.add_argument("--alert-score-trigger", choices=["above", "below"], default="above")
    parser.add_argument("--rate-limit", type=int, default=10)
    parser.add_argument("--rate-window", type=int, default=60)
    parser.add_argument("--rate-scope", choices=["global", "user"], default="global")

    args = parser.parse_args()

    # ── Load model ────────────────────────────────────────────────────────────
    from models.predict import IsolationForestPredictor

    try:
        predictor = IsolationForestPredictor(model_path=args.model_path)
        print(f"Loaded predictor; feature_order={predictor.feature_order}")
    except Exception as exc:
        print(f"[ERROR] Failed to load model: {exc}", file=sys.stderr)
        sys.exit(1)

    # ── Alert manager (optional) ──────────────────────────────────────────────
    alert_manager = None
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
        print(f"[WARN] AlertManager unavailable: {exc}")

    # ── Kafka output producer (both source modes can publish scored-events) ───
    kafka_out = None
    if args.source == "kafka":
        print(f"Connecting Kafka output producer to {args.bootstrap_servers} …")
        try:
            kafka_out = _build_kafka_producer(args.bootstrap_servers)
        except Exception as exc:
            print(f"[WARN] Could not create Kafka output producer: {exc}")

    pipeline = FeaturePipeline()

    # ── Run ───────────────────────────────────────────────────────────────────
    processed = 0

    if args.source == "kafka":
        print(f"Connecting Kafka consumer to {args.bootstrap_servers}, topic={args.input_topic} …")
        try:
            consumer = _build_kafka_consumer(args.bootstrap_servers, args.input_topic, args.group_id)
        except Exception as exc:
            print(f"[ERROR] Kafka consumer connection failed: {exc}", file=sys.stderr)
            print("Tip: run `docker compose up -d` first, or use --source generator", file=sys.stderr)
            sys.exit(1)

        print(f"Consumer ready. Waiting for events on '{args.input_topic}' …", flush=True)
        try:
            for msg in consumer:
                event = msg.value
                process_one(
                    event, pipeline, predictor, alert_manager,
                    args.threshold, args.db_path, args.fill_missing_with_zero,
                    kafka_out, args.output_topic,
                )
                processed += 1
        except KeyboardInterrupt:
            print("\nKeyboardInterrupt — shutting down consumer.")
        finally:
            consumer.close()

    else:  # generator mode
        gen = EventGenerator(seed=args.seed)
        # Cap at 50 events in generator mode (safety limit for tests/demos)
        count = max(1, min(args.count, 50))
        print(f"Starting generator mode: count={count}, interval={args.interval}s")
        try:
            while processed < count:
                event = gen.generate_transaction()
                process_one(
                    event, pipeline, predictor, alert_manager,
                    args.threshold, args.db_path, args.fill_missing_with_zero,
                    kafka_out, args.output_topic,
                )
                processed += 1
                time.sleep(max(0.0, float(args.interval)))
        except KeyboardInterrupt:
            print("\nKeyboardInterrupt — shutting down consumer.")

    if kafka_out is not None:
        kafka_out.flush()
        kafka_out.close()

    print(f"Consumer stopped after processing {processed} events.")


if __name__ == "__main__":
    main()
