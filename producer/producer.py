"""Simple producer script that continuously generates and prints transactions.

- Initializes `EventGenerator` once
- Continuously generates events in a loop and prints them (JSON)
- Controls event rate via `time.sleep(interval)`
- Supports graceful shutdown with KeyboardInterrupt (Ctrl+C)

Usage:
    python producer/producer.py --interval 1.0

Keep the script minimal and dependency-free (uses stdlib only).
"""
from __future__ import annotations

import argparse
import json
import time
from typing import Optional

try:
    # Prefer package-style import when available (run as module)
    from producer.event_generator import EventGenerator
except Exception:
    # When executed directly as a script (python producer/producer.py),
    # package imports may fail; add project root to sys.path and retry.
    import os
    import sys

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    # Load the module directly from file location to avoid package import issues
    import importlib.util

    module_path = os.path.join(repo_root, "producer", "event_generator.py")
    spec = importlib.util.spec_from_file_location("producer.event_generator", module_path)
    if spec is None or spec.loader is None:
        raise ImportError("Could not load producer.event_generator module")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    EventGenerator = mod.EventGenerator


def main() -> None:
    parser = argparse.ArgumentParser(description="Continuous synthetic transaction producer")
    parser.add_argument("--interval", type=float, default=1.0, help="Seconds to sleep between events (float)")
    parser.add_argument("--seed", type=int, default=None, help="Optional RNG seed for deterministic output")
    parser.add_argument("--count", type=int, default=0, help="Number of events to emit (0 means infinite)")
    args = parser.parse_args()

    # Initialize generator once (fast, lightweight)
    gen = EventGenerator(seed=args.seed)

    print(f"Starting producer: interval={args.interval}s, count={'infinite' if args.count<=0 else args.count}")
    emitted = 0

    try:
        while args.count <= 0 or emitted < args.count:
            event = gen.generate_transaction()
            # Print JSON for readability / downstream piping
            print(json.dumps(event, ensure_ascii=False))
            emitted += 1
            time.sleep(max(0.0, float(args.interval)))
    except KeyboardInterrupt:
        print("\nKeyboardInterrupt received — shutting down producer gracefully.")
    finally:
        print(f"Producer stopped after emitting {emitted} events.")


if __name__ == "__main__":
    main()
