"""Synthetic transaction producer.

Generates real-time transaction events and publishes them to either:
  - Apache Kafka topic `transactions` (default — acts as a real financial feed)
  - stdout as JSON (fallback for offline demos, use --sink stdout)

Usage:
    python producer/producer.py                        # Kafka (default)
    python producer/producer.py --sink stdout          # stdout only
    python producer/producer.py --interval 0.5 --count 100
"""
from __future__ import annotations

import argparse
import json
import sys
import os
import time
from typing import Optional

# Robust import so the script works both as a module and as a direct script
try:
    from producer.event_generator import EventGenerator
    import config
except Exception:
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    from producer.event_generator import EventGenerator
    import config


def _build_kafka_producer(bootstrap_servers: str):
    """Construct and return a KafkaProducer (lazy import — not needed for stdout mode)."""
    from kafka import KafkaProducer  # type: ignore[import]

    return KafkaProducer(
        bootstrap_servers=bootstrap_servers,
        value_serializer=lambda v: json.dumps(v, ensure_ascii=False).encode("utf-8"),
        key_serializer=lambda k: k.encode("utf-8") if k else None,
        # Reasonable defaults for a low-latency demo
        linger_ms=5,
        acks="all",
        retries=3,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Synthetic transaction producer")
    parser.add_argument(
        "--sink",
        choices=["kafka", "stdout"],
        default="kafka",
        help="Where to send events: 'kafka' (default) or 'stdout'",
    )
    parser.add_argument(
        "--bootstrap-servers",
        default=config.KAFKA_BOOTSTRAP_SERVERS,
        help=f"Kafka bootstrap servers (default: {config.KAFKA_BOOTSTRAP_SERVERS})",
    )
    parser.add_argument(
        "--topic",
        default=config.TOPIC_TRANSACTIONS,
        help=f"Kafka topic to publish to (default: {config.TOPIC_TRANSACTIONS})",
    )
    parser.add_argument("--interval", type=float, default=1.0, help="Seconds between events")
    parser.add_argument("--seed", type=int, default=None, help="Optional RNG seed")
    parser.add_argument("--count", type=int, default=0, help="Events to emit (0 = infinite)")
    args = parser.parse_args()

    gen = EventGenerator(seed=args.seed)

    # ── Build the appropriate sink ─────────────────────────────────────────────
    kafka_producer = None
    if args.sink == "kafka":
        print(f"Connecting to Kafka at {args.bootstrap_servers}, topic={args.topic} …", flush=True)
        try:
            kafka_producer = _build_kafka_producer(args.bootstrap_servers)
            print("Kafka producer ready.", flush=True)
        except Exception as exc:
            print(f"[ERROR] Could not connect to Kafka: {exc}", file=sys.stderr)
            print("Tip: run `docker compose up -d` first, or use --sink stdout", file=sys.stderr)
            sys.exit(1)

    def emit(event: dict) -> None:
        if kafka_producer is not None:
            kafka_producer.send(args.topic, key=event.get("user_id"), value=event)
        else:
            print(json.dumps(event, ensure_ascii=False), flush=True)

    print(
        f"Starting producer: sink={args.sink}, interval={args.interval}s, "
        f"count={'infinite' if args.count <= 0 else args.count}",
        flush=True,
    )
    emitted = 0

    try:
        while args.count <= 0 or emitted < args.count:
            event = gen.generate_transaction()
            emit(event)
            emitted += 1
            if args.sink == "stdout":
                # Echo to console in stdout mode so the user sees events
                pass  # already printed inside emit
            time.sleep(max(0.0, float(args.interval)))
    except KeyboardInterrupt:
        print("\nKeyboardInterrupt — shutting down producer gracefully.", flush=True)
    finally:
        if kafka_producer is not None:
            kafka_producer.flush()
            kafka_producer.close()
        print(f"Producer stopped after emitting {emitted} events.", flush=True)


if __name__ == "__main__":
    main()
