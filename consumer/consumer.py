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

    args = parser.parse_args()

    gen = EventGenerator(seed=args.seed)
    pipeline = FeaturePipeline()

    print(f"Starting consumer: interval={args.interval}s, count={'infinite' if args.count<=0 else args.count}")
    processed = 0

    try:
        while args.count <= 0 or processed < args.count:
            event = gen.generate_transaction()
            features = pipeline.process_event(event)

            # Print raw event and computed features (JSON)
            print("EVENT:", json.dumps(event, ensure_ascii=False))
            print("FEATURES:", json.dumps(features, ensure_ascii=False))

            processed += 1
            time.sleep(max(0.0, float(args.interval)))
    except KeyboardInterrupt:
        print("\nKeyboardInterrupt received — shutting down consumer gracefully.")
    finally:
        print(f"Consumer stopped after processing {processed} events.")


if __name__ == "__main__":
    main()
