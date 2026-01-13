from __future__ import annotations

import os
from typing import Iterable, Optional, Tuple

import numpy as np
from joblib import load

# Module-level cache for the loaded model to avoid re-loading on each call
_MODEL = None
# Track the path of the cached model so callers can request a different model
_MODEL_PATH: Optional[str] = None

# Default path points to top-level `models/isolation_forest.joblib` (local relative path)
_DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "isolation_forest.joblib")


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
	threshold: float = 0.18,
	model_path: Optional[str] = None,
) -> Tuple[float, bool]:
	"""Predict anomaly score and flag for a single feature vector.

	This function accepts a 1-D iterable (list / numpy array) representing
	the features for one sample and returns a tuple `(anomaly_score, is_anomaly)`.

	Implementation notes:
	- We use `IsolationForest.score_samples` which returns a score where larger
	  values generally indicate more normal points. We negate that value to
	  produce `anomaly_score` where **larger means more anomalous**. The
	  default threshold is positive (e.g., 0.18) which is chosen heuristically
	  to reduce false positives while allowing rare high-risk events to be
	  flagged; this positive threshold reflects the fact that `anomaly_score`
	  is the negated `score_samples` (so larger => more anomalous).

	Args:
		feature_vector: 1-D iterable of numeric features for a single sample.
		threshold: Numeric threshold on the returned `anomaly_score`. Samples
			with `anomaly_score >= threshold` are considered anomalous.
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

	# Ensure model exposes score_samples (IsolationForest does)
	if not hasattr(model, "score_samples"):
		raise AttributeError("Loaded model does not implement `score_samples`")

	# score_samples: higher => more normal. We negate to produce anomaly-oriented
	# score where larger => more anomalous.
	raw_score = model.score_samples(x)
	anomaly_score = float(-raw_score.item())

	# Log for transparency (both values shown)
	print(f"anomaly_score: {anomaly_score}, threshold: {threshold}")

	# Determine boolean flag: anomalous when anomaly_score >= threshold
	is_anomaly = anomaly_score >= float(threshold)

	return anomaly_score, is_anomaly


class IsolationForestPredictor:
    """Lightweight predictor wrapper around a joblib-saved IsolationForest.

    Supports predicting from either a numeric feature vector (list/ndarray)
    or a feature dict. When predicting from a dict, the predictor will build
    the input vector strictly in the order recorded on the trained model
    (the model's `feature_names_in_` attribute). If the loaded model does
    not include feature names, an explicit `feature_order` may be provided
    at construction. Passing dicts that omit required features will raise
    a clear KeyError.

    The anomaly score returned is `-decision_function(x)` so that larger
    values indicate more anomalous points (intuitive interpretation).
    """

    def __init__(self, model_path: Optional[str] = None, feature_order: Optional[list] = None):
        self.model_path = model_path
        # Force reload when a specific `model_path` is provided to avoid returning
        # a previously cached model that may come from a different file.
        self._model = _load_model(model_path, force_reload=(model_path is not None))

        # Prefer to check for `score_samples` which we use for scoring; fall back
        # to `decision_function` if `score_samples` is not available on the model.
        if not (hasattr(self._model, "score_samples") or hasattr(self._model, "decision_function")):
            raise AttributeError("Loaded model does not implement `score_samples` or `decision_function`")

        # Determine feature order: prefer the model's recorded feature names (if present)
        model_feats = getattr(self._model, "feature_names_in_", None)
        # Respect an explicit feature_order parameter if provided (allows using a model without recorded feature names)
        if feature_order is not None:
            self.feature_order = list(feature_order)
        elif model_feats is not None:
            # Use exact feature ordering from the trained model
            self.feature_order = list(model_feats)
        else:
            # Leave unset if neither is available; dict-based prediction will raise until feature_order is set
            self.feature_order = None

        # Validate that provided/derived feature_order matches model's expected feature count (if available)
        model_n_feats = getattr(self._model, "n_features_in_", None)
        if model_n_feats is None and model_feats is not None:
            # Try to infer from model_feats
            model_n_feats = len(model_feats)

        if self.feature_order is not None and model_n_feats is not None and len(self.feature_order) != int(model_n_feats):
            raise ValueError(f"Feature order length {len(self.feature_order)} does not match model's expected feature count {model_n_feats}")

    def _dict_to_dataframe(self, features: Dict[str, object]):
        """Convert a feature dict into a pandas DataFrame with exact column order.

        Behavior:
        - Uses `self.feature_order` (set from model.feature_names_in_ if available)
        - Raises KeyError listing missing features if any are absent
        - Raises ValueError if values are None or cannot be converted to float
        - Returns a single-row `pandas.DataFrame` with columns in `self.feature_order`
        """
        if self.feature_order is None:
            raise ValueError("feature_order is not set; cannot convert dict to DataFrame")

        # Check for missing feature names and raise a clear KeyError
        missing = [name for name in self.feature_order if name not in features]
        if missing:
            raise KeyError(f"Missing required feature(s): {missing}")

        # Build an ordered list of numeric values, validating conversions explicitly
        ordered_vals = []
        for name in self.feature_order:
            val = features[name]
            if val is None:
                raise ValueError(f"Feature '{name}' is None; expected numeric value")
            try:
                num = float(val)
            except Exception as exc:
                raise ValueError(f"Feature '{name}' with value {val!r} cannot be converted to float") from exc
            ordered_vals.append(num)

        # Create a single-row DataFrame with the exact column names and preserved order
        import pandas as pd

        df = pd.DataFrame([ordered_vals], columns=self.feature_order)
        return df

    def predict(self, features: object, threshold: float = 0.18) -> Tuple[float, bool]:
        """Predict anomaly score and boolean flag for a single sample.

        Args:
            features: Either a 1-D iterable of numeric features (list/ndarray)
                or a dict mapping feature names to values.
            threshold: Numeric threshold on returned `anomaly_score`. Points with
                `anomaly_score >= threshold` are considered anomalous. Default is 0.18.

        Returns:
            (anomaly_score: float, is_anomaly: bool)
        """
        # Convert dict -> single-row DataFrame if needed (ensures feature names are explicit)
        if isinstance(features, dict):
            X = self._dict_to_dataframe(features)
        else:
            x = np.asarray(features, dtype=float)
            if x.ndim != 1:
                raise ValueError("feature vector must be 1-D for a single sample")
            # If we have a known feature_order, convert to DataFrame with exact column names
            if self.feature_order is not None:
                if x.size != len(self.feature_order):
                    raise ValueError(f"Feature vector length {x.size} does not match model feature count {len(self.feature_order)}")
                import pandas as pd
                X = pd.DataFrame([x.tolist()], columns=self.feature_order)
            else:
                # Fall back to passing a numpy array if no feature names are available
                X = x.reshape(1, -1)

        if not hasattr(self._model, "score_samples"):
            raise AttributeError("Loaded model does not implement `score_samples`")

        # score_samples: higher => more normal. Negate to make anomaly-oriented
        # score where larger means more anomalous. The default threshold is
        # negative (e.g. -0.18) to reduce false positives while allowing rare
        # high-risk events to be flagged.
        raw = self._model.score_samples(X)
        anomaly_score = float(-raw.item())

        # Log anomaly score and threshold for visibility
        print(f"anomaly_score: {anomaly_score}, threshold: {threshold}")

        # Points with anomaly_score >= threshold are considered anomalous
        is_anom = bool(anomaly_score >= float(threshold))
        return anomaly_score, is_anom


# Backwards-compatible convenience function
def predict_anomaly(
    feature_vector: Iterable[float],
    threshold: float = 0.45,
    model_path: Optional[str] = None,
) -> Tuple[float, bool]:
    """Deprecated convenience wrapper: predict from a numeric feature vector.

    Prefer using `IsolationForestPredictor` for dict inputs or when needing
    explicit feature ordering. This wrapper keeps the previous function's
    semantics but uses the same default threshold as `IsolationForestPredictor`.
    """
    predictor = IsolationForestPredictor(model_path=model_path)
    return predictor.predict(feature_vector, threshold=threshold)


__all__ = [
    "IsolationForestPredictor",
    "predict_anomaly",
    "_load_model",
    "_DEFAULT_MODEL_PATH",
]


