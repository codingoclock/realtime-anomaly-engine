"""Unit tests for the IsolationForest predictor wrapper."""
from __future__ import annotations

import joblib
import numpy as np
from pathlib import Path
from sklearn.ensemble import IsolationForest
import pandas as pd

from models.predict import IsolationForestPredictor


def train_and_dump_model(tmp_path: Path, n_features: int = 2):
    # Create a simple training dataset using pandas so feature_names_in_ is set
    df = pd.DataFrame(np.random.RandomState(0).randn(100, n_features), columns=[f"f{i}" for i in range(n_features)])
    model = IsolationForest(random_state=0)
    model.fit(df)
    path = tmp_path / "if_model.joblib"
    joblib.dump(model, path)
    return str(path), [f"f{i}" for i in range(n_features)]


def test_predict_from_vector(tmp_path: Path):
    path, feature_names = train_and_dump_model(tmp_path, n_features=2)
    predictor = IsolationForestPredictor(model_path=path)

    vec = [0.0, 0.0]
    score, flag = predictor.predict(vec, threshold=0.0)

    # Compute expected raw and assert sign interpretation
    model = joblib.load(path)
    raw = model.decision_function([vec])
    expected_score = -float(raw.item())
    assert np.isclose(score, expected_score)
    assert flag == (score >= 0.0)


def test_predict_from_dict_with_model_feature_names(tmp_path: Path):
    path, feature_names = train_and_dump_model(tmp_path, n_features=3)
    predictor = IsolationForestPredictor(model_path=path)

    # Input dict with shuffled keys
    sample = {"f2": 0.1, "f0": -0.2, "f1": 0.3}
    score, flag = predictor.predict(sample, threshold=1e9)  # use extreme threshold -> should be False
    assert flag is False

    # Ensure vectorization follows model.feature_names_in_
    v = np.asarray([sample[n] for n in predictor.feature_order], dtype=float)
    model = joblib.load(path)
    expected = -float(model.decision_function(np.atleast_2d(v)).item())
    assert np.isclose(score, expected)


def test_predict_from_dict_with_explicit_order(tmp_path: Path):
    path, _ = train_and_dump_model(tmp_path, n_features=2)
    # Provide explicit feature order not present in model
    predictor = IsolationForestPredictor(model_path=path, feature_order=["a", "b"])
    sample = {"a": 1.0, "b": 2.0}
    score, flag = predictor.predict(sample)
    # no exception -> success; check types
    assert isinstance(score, float)
    assert isinstance(flag, bool)


def test_missing_feature_raises(tmp_path: Path):
    path, _ = train_and_dump_model(tmp_path, n_features=2)
    predictor = IsolationForestPredictor(model_path=path, feature_order=["x", "y"])  # expects x,y
    try:
        predictor.predict({"x": 1.0})
        assert False, "Expected KeyError for missing feature"
    except KeyError:
        pass
