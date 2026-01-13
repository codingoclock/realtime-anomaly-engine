"""Train and save an IsolationForest model from generated features.

Usage:
    python models/train_model.py [--count N] [--seed SEED] [--output PATH]

Behavior:
- Imports `generate_feature_dataframe` from `models.train_dataset` to build data
- Trains an IsolationForest with n_estimators=200, contamination=0.01, random_state=42
- Ensures the model stores the exact feature order used for training
- Saves the trained model to `models/isolation_forest.joblib` by default
- Prints number of samples and the feature order
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

# Make project root importable when running this script from models/ directory
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from joblib import dump
import numpy as np
from sklearn.ensemble import IsolationForest

from models.train_dataset import generate_feature_dataframe


DEFAULT_OUTPUT = SCRIPT_DIR / "isolation_forest.joblib"


def train_and_save(count: int = 5000, seed: int | None = None, output: str | Path = DEFAULT_OUTPUT) -> None:
    df = generate_feature_dataframe(count=count, seed=seed)

    if df.empty:
        raise RuntimeError("No training data available after cleaning; aborting training")

    feature_cols = list(df.columns)

    X = df[feature_cols]

    # Create and train IsolationForest with requested fixed hyperparameters
    model = IsolationForest(n_estimators=200, contamination=0.01, random_state=42)
    model.fit(X)

    # Ensure the model stores the exact feature order used for training
    # scikit-learn may set `feature_names_in_` when fitting with DataFrame, but set explicitly for clarity
    try:
        model.feature_names_in_ = np.asarray(feature_cols, dtype=object)
    except Exception:
        # Fallback: attach attribute directly
        setattr(model, "feature_names_in_", np.asarray(feature_cols, dtype=object))

    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    dump(model, out_path)

    print(f"Trained IsolationForest on {len(X)} samples")
    print(f"Saved model to: {out_path}")
    print(f"Feature order saved in model: {feature_cols}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train IsolationForest model from generated features")
    parser.add_argument("--count", type=int, default=5000, help="Number of synthetic events to generate (default: 5000)")
    parser.add_argument("--seed", type=int, default=None, help="Optional RNG seed for reproducibility")
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT), help="Output model path")
    args = parser.parse_args()

    train_and_save(count=args.count, seed=args.seed, output=args.output)


if __name__ == "__main__":
    main()
