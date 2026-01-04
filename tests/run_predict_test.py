"""Simple test runner for models.predict.predict_anomaly function to verify setup.
"""
from __future__ import annotations

from models.predict import predict_anomaly, _DEFAULT_MODEL_PATH, _load_model


def main():
    try:
        # Load the model to inspect expected number of features
        model = _load_model()
        n_features = getattr(model, "n_features_in_", None)
        if n_features is None:
            # Fall back to 4 features if attribute not present
            n_features = 4

        # Construct a neutral sample with the correct dimensionality
        sample = [0.0] * int(n_features)
        score, is_anom = predict_anomaly(sample)
        print(f"Model path: {_DEFAULT_MODEL_PATH}")
        print(f"Anomaly score: {score:.6f}")
        print(f"Is anomaly: {is_anom}")
    except Exception as exc:
        print("Prediction failed:", type(exc).__name__, exc)


if __name__ == "__main__":
    main()
