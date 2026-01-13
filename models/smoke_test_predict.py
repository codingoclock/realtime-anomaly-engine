"""Smoke test: generate one event, compute features, and predict anomaly.

Usage:
    python models/smoke_test_predict.py [--seed N] [--model-path PATH] [--threshold FLOAT]

Exits with non-zero code if prediction cannot be made or returned score is None.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Ensure project root is on sys.path when run from any cwd
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from producer.event_generator import EventGenerator
from consumer.feature_pipeline import FeaturePipeline
from models.predict import IsolationForestPredictor


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke test for feature pipeline + predictor")
    parser.add_argument("--seed", type=int, default=None, help="RNG seed for event generation")
    parser.add_argument("--model-path", type=str, default=None, help="Optional path to joblib model")
    parser.add_argument("--threshold", type=float, default=0.0, help="Anomaly score threshold")
    args = parser.parse_args()

    gen = EventGenerator(seed=args.seed)
    pipeline = FeaturePipeline()

    event = gen.generate_transaction()
    features = pipeline.process_event(event)

    print("FEATURES:", json.dumps(features, ensure_ascii=False))

    try:
        predictor = IsolationForestPredictor(model_path=args.model_path)
    except Exception as exc:
        print("ERROR: could not load predictor:", type(exc).__name__, exc)
        return 2

    try:
        score, is_anomaly = predictor.predict(features, threshold=args.threshold)
    except Exception as exc:
        print("ERROR: prediction failed:", type(exc).__name__, exc)
        return 3

    print(f"ANOMALY_SCORE: {score}")
    print(f"IS_ANOMALY: {is_anomaly}")

    # Strict validation: never allow None
    if score is None:
        print("ERROR: anomaly_score is None")
        return 4
    if is_anomaly is None:
        print("ERROR: is_anomaly is None")
        return 5

    # Ensure correct types
    if not isinstance(score, (float, int)):
        print(f"ERROR: anomaly_score is not numeric: {type(score).__name__}")
        return 6
    if not isinstance(is_anomaly, bool):
        print(f"ERROR: is_anomaly is not bool: {type(is_anomaly).__name__}")
        return 7

    print("Smoke test passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
