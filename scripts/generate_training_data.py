"""Generate a CSV dataset of computed features from synthetic events.

Usage:
    python scripts/generate_training_data.py [--count N] [--seed SEED]

The script will:
- Initialize EventGenerator and FeaturePipeline
- Generate N events (default 500)
- Collect only the computed feature dictionaries
- Save the dataset as CSV at `data/training_features.csv`
- Ensure all values written are numeric
- Print how many rows were written
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

# Make project root importable when running script from scripts/ directory
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from producer.event_generator import EventGenerator
from consumer.feature_pipeline import FeaturePipeline


def generate_dataset(count: int = 500, seed: int | None = None, output: str | Path = "data/training_features.csv") -> int:
    """Generate `count` events, compute features and write them to `output`.

    Returns the number of rows written.
    """
    gen = EventGenerator(seed=seed)
    pipeline = FeaturePipeline()

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        'transaction_amount',
        'rolling_mean_amount_per_user',
        'transaction_count_last_1_min',
        'time_since_last_transaction_seconds',
    ]

    written = 0
    with output_path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for _ in range(int(count)):
            event = gen.generate_transaction()
            features = pipeline.process_event(event)

            # Ensure all features are numeric and not None
            for k in fieldnames:
                if k not in features:
                    raise RuntimeError(f"Missing feature '{k}' in computed features")
                v = features[k]
                if v is None:
                    raise RuntimeError(f"Feature '{k}' is None for event {event.get('transaction_id')}" )
                if not isinstance(v, (int, float)):
                    # Attempt a safe conversion
                    try:
                        v = float(v)
                    except Exception as exc:  # pragma: no cover - defensive
                        raise RuntimeError(f"Feature '{k}' is not numeric and cannot be converted: {v!r}") from exc
                # Normalize ints to int and floats to float (round floats to 3 decimals where applicable)
                if isinstance(v, float):
                    # round transaction_time to 3 decimals to match pipeline behavior
                    if k == 'time_since_last_transaction_seconds':
                        v = round(v, 3)
                    else:
                        v = float(v)
                elif isinstance(v, int):
                    v = int(v)

                features[k] = v

            writer.writerow({k: features[k] for k in fieldnames})
            written += 1

    print(f"Wrote {written} rows to {output_path}")

    # --- Validation: load with pandas and ensure no NaNs ---
    try:
        import pandas as pd
    except Exception as exc:  # pragma: no cover - defensive
        raise RuntimeError("pandas is required for CSV validation but is not installed") from exc

    df = pd.read_csv(output_path)
    # Print summary info
    print(f"Validation: rows={len(df)}")
    print(f"Validation: columns={list(df.columns)}")
    nan_counts = df.isna().sum()
    print("Validation: NaN counts per column:\n" + str(nan_counts.to_dict()))

    # Raise if any NaNs present
    total_nans = int(nan_counts.sum())
    if total_nans > 0:
        raise RuntimeError(f"Validation failed: {total_nans} NaN values found in {output_path}")

    print("Validation passed: no NaN values found")
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate training feature CSV from synthetic events")
    parser.add_argument("--count", type=int, default=500, help="Number of events to generate (default: 500)")
    parser.add_argument("--seed", type=int, default=None, help="RNG seed for reproducibility")
    parser.add_argument("--output", type=str, default="data/training_features.csv", help="Output CSV path")
    args = parser.parse_args()

    generate_dataset(count=args.count, seed=args.seed, output=args.output)


if __name__ == "__main__":
    main()
