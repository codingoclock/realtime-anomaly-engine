from __future__ import annotations

import os
from typing import Iterable, Optional, Tuple

import numpy as np
from joblib import load

# Module-level cache for the loaded model to avoid re-loading on each call
_MODEL = None
# Track the path of the cached model so callers can request a different model
_MODEL_PATH: Optional[str] = None

# Default path points to models/models/isolation_forest.joblib relative to this file
_DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "isolation_forest.joblib")


def _load_model(path: Optional[str] = None, force_reload: bool = False):
    """Load and cache the joblib-saved Isolation Forest model.

    Args:
        path: Optional filesystem path to the .joblib model. If None, uses
            the module default `_DEFAULT_MODEL_PATH`.
        force_reload: If True, re-load the model even if cached.

    Returns:
        The loaded model object.
    """
    global _MODEL, _MODEL_PATH
    model_path = path or _DEFAULT_MODEL_PATH

    # Reload when requested explicitly, or when requesting a different model
    if _MODEL is None or force_reload or (_MODEL_PATH is not None and _MODEL_PATH != model_path):
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"Isolation Forest model not found at: {model_path}")
        # joblib.load is fast for sklearn models and suitable for production
        _MODEL = load(model_path)
        _MODEL_PATH = model_path
    return _MODEL

def predict_anomaly(
	feature_vector: Iterable[float],
	threshold: float = 0.0,
	model_path: Optional[str] = None,
) -> Tuple[float, bool]:
	"""Predict anomaly score and flag for a single feature vector.

	This function accepts a 1-D iterable (list / numpy array) representing
	the features for one sample and returns a tuple `(anomaly_score, is_anomaly)`.

	Implementation notes:
	- We use `IsolationForest.decision_function` which returns higher values
	  for more normal points. To present a more intuitive "anomaly score"
	  where larger means more anomalous, we negate the decision function.
	  Thus: `anomaly_score = -decision_function(sample)`.
	- The `threshold` is compared against `anomaly_score`: if
	  `anomaly_score >= threshold` the sample is marked anomalous.
	  Default `threshold=0.0` corresponds to `decision_function <= 0`.

	Args:
		feature_vector: 1-D iterable of numeric features for a single sample.
		threshold: Numeric threshold on the returned `anomaly_score` used to
			determine the boolean `is_anomaly` flag. Higher threshold -> fewer
			points classified as anomalous.
		model_path: Optional path to the .joblib model; if omitted the module
			default path is used and the loaded model is cached.

	Returns:
		(anomaly_score, is_anomaly)

	Raises:
		FileNotFoundError: If the model file cannot be found.
		ValueError: If the input `feature_vector` is not one-dimensional.
	"""
	# Convert input to numpy array (fast) and ensure shape (1, n_features)
	x = np.asarray(feature_vector, dtype=float)
	if x.ndim != 1:
		raise ValueError("feature_vector must be a 1-D iterable for a single sample")
	x = x.reshape(1, -1)

	# Load (and cache) the model
	model = _load_model(model_path)

	# Ensure model exposes decision_function (IsolationForest does)
	if not hasattr(model, "decision_function"):
		raise AttributeError("Loaded model does not implement `decision_function`")

	# decision_function: higher => more normal. Negate -> higher => more anomalous.
	# Use item() to return a Python float instead of numpy scalar.
	raw_score = model.decision_function(x)
	anomaly_score = float(-raw_score.item())

	# Determine boolean flag based on provided threshold
	is_anomaly = anomaly_score >= float(threshold)

	return anomaly_score, is_anomaly


class IsolationForestPredictor:
    """Lightweight predictor wrapper around a joblib-saved IsolationForest.

    Supports predicting from either a numeric feature vector (list/ndarray)
    or a feature dict. When given a dict, features are ordered according to
    `feature_order` (list of names). If `feature_order` is omitted, the
    predictor will try to use the model's `feature_names_in_` attribute if
    available (trained using a pandas.DataFrame). Otherwise the caller must
    provide `feature_order`.

    The anomaly score returned is `-decision_function(x)` so that larger
    values indicate more anomalous points (intuitive interpretation).
    """

    def __init__(self, model_path: Optional[str] = None, feature_order: Optional[list] = None):
        self.model_path = model_path
        # Force reload when a specific `model_path` is provided to avoid returning
        # a previously cached model that may come from a different file.
        self._model = _load_model(model_path, force_reload=(model_path is not None))

        if not hasattr(self._model, "decision_function"):
            raise AttributeError("Loaded model does not implement `decision_function`")

        # Determine default feature order from model if available (trained with DataFrame)
        model_feats = getattr(self._model, "feature_names_in_", None)
        self.feature_order = list(model_feats) if (feature_order is None and model_feats is not None) else feature_order

    def _dict_to_vector(self, features: Dict[str, object]) -> np.ndarray:
        """Convert a feature dict into a numeric vector in the configured order.

        Raises:
            ValueError: if `feature_order` is not set or required keys are missing.
        """
        if self.feature_order is None:
            raise ValueError("feature_order is not set; cannot convert dict to vector")
        try:
            vec = [float(features[name]) for name in self.feature_order]
        except KeyError as exc:
            raise KeyError(f"Missing feature in input dict: {exc.args[0]}") from exc
        return np.asarray(vec, dtype=float)

    def predict(self, features: object, threshold: float = 0.0) -> Tuple[float, bool]:
        """Predict anomaly score and boolean flag for a single sample.

        Args:
            features: Either a 1-D iterable of numeric features (list/ndarray)
                or a dict mapping feature names to values.
            threshold: Numeric threshold on returned `anomaly_score`. Larger
                threshold -> fewer samples flagged anomalous.

        Returns:
            (anomaly_score: float, is_anomaly: bool)
        """
        # Convert dict -> vector if needed
        if isinstance(features, dict):
            x = self._dict_to_vector(features)
        else:
            x = np.asarray(features, dtype=float)
            if x.ndim != 1:
                raise ValueError("feature vector must be 1-D for a single sample")

        x = x.reshape(1, -1)

        if not hasattr(self._model, "decision_function"):
            raise AttributeError("Loaded model does not implement `decision_function`")

        raw = self._model.decision_function(x)
        anomaly_score = float(-raw.item())
        is_anom = anomaly_score >= float(threshold)
        return anomaly_score, is_anom


# Backwards-compatible convenience function
def predict_anomaly(
    feature_vector: Iterable[float],
    threshold: float = 0.0,
    model_path: Optional[str] = None,
) -> Tuple[float, bool]:
    """Deprecated convenience wrapper: predict from a numeric feature vector.

    Prefer using `IsolationForestPredictor` for dict inputs or when needing
    explicit feature ordering. This wrapper keeps the previous function's
    semantics.
    """
    predictor = IsolationForestPredictor(model_path=model_path)
    return predictor.predict(feature_vector, threshold=threshold)


__all__ = [
    "IsolationForestPredictor",
    "predict_anomaly",
    "_load_model",
    "_DEFAULT_MODEL_PATH",
]


