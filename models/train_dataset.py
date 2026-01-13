"""Generate a cleaned training DataFrame from synthetic events.

Provides:
- generate_feature_dataframe(count=5000, seed=None) -> pd.DataFrame

Behavior:
- Uses `EventGenerator` and `FeaturePipeline` to generate `count` events
- Collects only the feature dictionaries returned by the pipeline
- Drops rows with any None/NaN values
- Returns the resulting pandas.DataFrame

When run as a script, prints the final DataFrame shape.
"""
from __future__ import annotations

from pathlib import Path
import sys
from typing import List, Dict, Any, Optional

# Make project root importable when running from models/ dir
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from producer.event_generator import EventGenerator
from consumer.feature_pipeline import FeaturePipeline


def generate_feature_dataframe(count: int = 5000, seed: Optional[int] = None) -> pd.DataFrame:
    """Generate `count` synthetic events, compute features, and return cleaned DataFrame.

    - Collects only feature dicts returned by `FeaturePipeline.process_event`
    - Drops rows with any None/NaN values
    - Ensures all columns are numeric
    """
    gen = EventGenerator(seed=seed)
    pipeline = FeaturePipeline()

    rows: List[Dict[str, Any]] = []

    for _ in range(int(count)):
        event = gen.generate_transaction()
        features = pipeline.process_event(event)
        rows.append(features)

    df = pd.DataFrame(rows)

    # Convert to numeric (coerce non-numeric -> NaN)
    df = df.apply(pd.to_numeric, errors="coerce")

    # Ensure `time_since_last_transaction_seconds` is present and never null
    if 'time_since_last_transaction_seconds' not in df.columns:
        raise RuntimeError("Expected feature 'time_since_last_transaction_seconds' missing from generated features")
    # Fill missing time deltas with 0.0 (first transactions)
    df['time_since_last_transaction_seconds'] = df['time_since_last_transaction_seconds'].fillna(0.0)

    # Logging: column names, null counts per column, and first 3 rows
    null_counts = df.isna().sum()
    print(f"Columns: {list(df.columns)}")
    print("Null counts per column:", dict(null_counts))
    print("First 3 rows:")
    if df.shape[0] > 0:
        print(df.head(3).to_string(index=False))
    else:
        print("<empty dataframe>")

    # Drop any remaining rows with NaN in other columns
    before = df.shape[0]
    df = df.dropna(axis=0, how="any").reset_index(drop=True)
    after = df.shape[0]

    if after < before:
        print(f"Dropped {before-after} rows with None/NaN values (after filling time_since_last_transaction_seconds)")

    # Final safety check: ensure time_since_last_transaction_seconds has no nulls
    if df['time_since_last_transaction_seconds'].isna().any():
        raise RuntimeError("time_since_last_transaction_seconds contains null values after cleaning")

    return df


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Generate a training DataFrame from synthetic events and print its shape")
    parser.add_argument("--count", type=int, default=5000, help="Number of synthetic events to generate (default: 5000)")
    parser.add_argument("--seed", type=int, default=None, help="Optional RNG seed for reproducibility")
    args = parser.parse_args()

    df = generate_feature_dataframe(count=args.count, seed=args.seed)
    print(f"Generated DataFrame with shape: {df.shape}")


if __name__ == "__main__":
    main()
