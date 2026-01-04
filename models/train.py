"""
Train an Isolation Forest anomaly detector on the Kaggle Credit Card Fraud dataset.

Notes:
- Expects the CSV file to be located at `data/raw/creditcard.csv` by default.
- Selects features V1..V28 and Amount (excludes the Class label).
- Saves the trained model to `models/isolation_forest.joblib` by default.

This script is intentionally simple and well-commented for beginners.
"""

from pathlib import Path
import argparse

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
import joblib


def load_data(csv_path: Path) -> pd.DataFrame:
    # If path doesn't exist as provided (absolute or relative to CWD), try resolving
    # it relative to the project root (two levels up from this file). This makes the
    # script runnable from the `models/` folder or the repository root.
    if not csv_path.exists():
        project_root = Path(__file__).resolve().parents[1]
        alt_path = project_root / csv_path
        if alt_path.exists():
            csv_path = alt_path
        else:
            raise FileNotFoundError(
                f"CSV file not found at {csv_path} (cwd: {Path.cwd()}).\n"
                f"Also tried: {alt_path} (project root). Please place the Kaggle CSV in one of these locations."
            )

    # Use pandas to read the CSV. This dataset is not huge, so read_csv is fine.
    df = pd.read_csv(csv_path)
    return df


def select_features(df: pd.DataFrame) -> pd.DataFrame:
    """Select the features V1..V28 and Amount from the dataset.

    The dataset also contains a `Class` column (0 = normal, 1 = fraud) which we must not
    use as a feature for unsupervised anomaly detection.
    """
    feature_cols = [f"V{i}" for i in range(1, 29)] + ["Amount"]

    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected feature columns from CSV: {missing}")

    X = df[feature_cols].astype(float)
    return X


def train_isolation_forest(X: pd.DataFrame, random_state: int = 42) -> IsolationForest:
    """Train an Isolation Forest on the provided features.

    Args:
        X: Feature matrix (DataFrame or array-like).
        random_state: Random seed for reproducibility.

    Returns:
        Trained IsolationForest model.
    """
    # Create the model. n_estimators and contamination can be tuned for production.
    model = IsolationForest(n_estimators=100, contamination='auto', random_state=random_state)

    # Fit the model on the feature matrix
    model.fit(X)
    return model


def compute_anomaly_scores(model: IsolationForest, X: pd.DataFrame) -> np.ndarray:
    """Compute anomaly scores for each sample.

    IsolationForest.score_samples returns an anomaly score where **higher** values mean
    more normal points. For convenience, we invert the sign so that **higher** values
    indicate more anomalous samples.
    """
    # score_samples: higher -> more normal; negate to get anomaly magnitude
    scores = -model.score_samples(X)
    return scores


def save_model(model: IsolationForest, out_path: Path) -> None:
    """Save the trained model to disk using joblib."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out_path)


def main(args: argparse.Namespace) -> None:
    csv_path = Path(args.csv)
    model_out = Path(args.out)

    print(f"Loading CSV from: {csv_path}")
    df = load_data(csv_path)
    print(f"Loaded dataframe with shape: {df.shape}")

    print("Selecting features V1..V28 and Amount (excluding Class)...")
    X = select_features(df)
    print(f"Feature matrix shape: {X.shape}")

    print("Training Isolation Forest...")
    model = train_isolation_forest(X, random_state=args.random_state)
    print("Training complete.")

    print("Computing anomaly scores for each sample...")
    df['anomaly_score'] = compute_anomaly_scores(model, X)
    print("Anomaly scores computed and added to dataframe as 'anomaly_score'.")

    print(f"Saving trained model to: {model_out}")
    save_model(model, model_out)
    print("Model saved.")

    if args.save_scores:
        out_scores = Path(args.save_scores)
        out_scores.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_scores, index=False)
        print(f"Saved dataframe with anomaly scores to: {out_scores}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train an Isolation Forest on credit card dataset and save the model")

    parser.add_argument(
        "--csv",
        type=str,
        default="data/raw/creditcard.csv",
        help="Path to the credit card CSV file (default: data/raw/creditcard.csv)",
    )

    parser.add_argument(
        "--out",
        type=str,
        default="models/isolation_forest.joblib",
        help="Output path to save the trained model (default: models/isolation_forest.joblib)",
    )

    parser.add_argument(
        "--save-scores",
        type=str,
        default="data/processed/predictions_with_scores.csv",
        help="Optional path to save the dataset augmented with anomaly scores (default: data/processed/predictions_with_scores.csv)",
    )

    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducible training (default: 42)",
    )

    args = parser.parse_args()
    main(args)


