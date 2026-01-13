"""Train an IsolationForest model on the generated feature CSV.

Usage:
    python models/train_local.py [--input PATH] [--output PATH] [--random-state N]

Behavior:
- Loads CSV (default: data/training_features.csv)
- Uses all columns as features (does not hardcode names)
- Validates values are numeric and that there are no NaNs
- Trains sklearn.IsolationForest with a fixed random_state (default: 42)
- Saves the trained model to `models/isolation_forest.joblib` by default
- Prints number of samples and the feature order used for training
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

# Make project root importable when running this script from models/ directory
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from joblib import dump
from sklearn.ensemble import IsolationForest


def train_model(input_csv: str | Path = "data/training_features.csv", output_model: str | Path = "models/isolation_forest.joblib", random_state: int = 42) -> None:
    input_path = Path(input_csv)
    if not input_path.is_file():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    df = pd.read_csv(input_path)

    if df.empty:
        raise ValueError("Input CSV contains no rows")

    # Use all columns as features (do not hardcode names)
    feature_cols = list(df.columns)

    # Ensure all columns are numeric and there are no NaNs
    try:
        X = df[feature_cols].apply(pd.to_numeric, errors="raise")
    except Exception as exc:  # pragma: no cover - validation
        raise ValueError("Failed to convert feature columns to numeric") from exc

    nan_counts = X.isna().sum().sum()
    if nan_counts > 0:
        raise ValueError(f"Found {int(nan_counts)} NaN values in feature CSV; cannot train")

    # Train IsolationForest using the DataFrame so scikit-learn records feature names
    model = IsolationForest(random_state=int(random_state))
    model.fit(X)

    # Ensure models directory exists
    out_path = Path(output_model)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    dump(model, out_path)

    print(f"Trained IsolationForest on {len(X)} samples")
    print(f"Feature order used for training: {feature_cols}")
    print(f"Saved model to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a local IsolationForest model from generated CSV")
    parser.add_argument("--input", type=str, default="data/training_features.csv", help="Input CSV path")
    parser.add_argument("--output", type=str, default="models/isolation_forest.joblib", help="Output model path")
    parser.add_argument("--random-state", type=int, default=42, help="Random state for reproducibility")
    args = parser.parse_args()

    train_model(input_csv=args.input, output_model=args.output, random_state=args.random_state)


if __name__ == "__main__":
    main()
