from __future__ import annotations

import os
from typing import Iterable, Optional, Tuple

import numpy as np
from joblib import load

# Module-level cache for the loaded model to avoid re-loading on each call
_MODEL = None

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
	global _MODEL
	if _MODEL is None or force_reload:
		model_path = path or _DEFAULT_MODEL_PATH
		if not os.path.isfile(model_path):
			raise FileNotFoundError(f"Isolation Forest model not found at: {model_path}")
		# joblib.load is fast for sklearn models and suitable for production
		_MODEL = load(model_path)
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


__all__ = ["predict_anomaly", "_load_model", "_DEFAULT_MODEL_PATH"]

