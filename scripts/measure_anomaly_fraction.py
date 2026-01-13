"""Measure anomaly fraction using current EventGenerator and saved model.

Usage: python scripts/measure_anomaly_fraction.py [num_events]
"""
from __future__ import annotations

import sys
import os
# Ensure workspace root is on PYTHONPATH when running as a script
ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from producer.event_generator import EventGenerator
from consumer.feature_pipeline import FeaturePipeline
from models.predict import IsolationForestPredictor


def main(n_events: int = 2000, seed: int = 42):
    gen = EventGenerator(seed=seed)
    pipeline = FeaturePipeline()
    predictor = IsolationForestPredictor()

    anomalies = 0
    scores = []
    samples = []
    for i in range(n_events):
        ev = gen.generate_transaction()
        feats = pipeline.process_event(ev)
        score, is_anom = predictor.predict(feats)
        scores.append(score)
        if is_anom:
            anomalies += 1
            if len(samples) < 5:
                samples.append((ev, feats, score))

    print(f"out of {n_events} events: {anomalies} anomalies, fraction: {anomalies / n_events:.4f}")
    if samples:
        print("\nSample anomalies (up to 5):")
        for ev, feats, score in samples:
            print(ev)
            print(feats)
            print('score:', score)
            print('-' * 40)


if __name__ == '__main__':
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
    main(n_events=n)
