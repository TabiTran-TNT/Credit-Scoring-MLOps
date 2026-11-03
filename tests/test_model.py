from hashlib import sha256

import joblib
import numpy as np
import pandas as pd
import pytest

from credit_scoring.features import prepare_features


def test_loader_preserves_feature_order_and_model_identity(runtime):
    from app import model_loader
    model = model_loader.load_model()
    assert model_loader.get_feature_columns() == list(model.feature_names_in_)
    expected = "sha256:" + sha256(runtime["model"].read_bytes()).hexdigest()
    assert model_loader.get_model_id() == expected


def test_loader_rejects_mismatched_feature_artifact(runtime, tmp_path, monkeypatch):
    from app import model_loader
    columns = joblib.load(runtime["columns"])
    path = tmp_path / "wrong-columns.pkl"
    joblib.dump(list(reversed(columns)), path)
    monkeypatch.setenv("FEATURES_PATH", str(path))
    with pytest.raises(ValueError, match="do not match"):
        model_loader.load_model()


def test_missing_model_prevents_startup(runtime, tmp_path, monkeypatch, read_events):
    from fastapi.testclient import TestClient
    from app.main import app
    monkeypatch.setenv("MODEL_PATH", str(tmp_path / "missing.pkl"))
    with pytest.raises(FileNotFoundError):
        with TestClient(app):
            pytest.fail("Startup must not succeed without a model")
    assert read_events()[-1]["event"] == "startup_error"


@pytest.mark.artifact
def test_actual_saved_model_can_score(valid_applicant):
    # Optional local smoke check; all other tests use temporary synthetic models.
    from conftest import PROJECT_ROOT
    model_path = PROJECT_ROOT / "model/credit_model.pkl"
    columns_path = PROJECT_ROOT / "model/feature_columns.pkl"
    if not model_path.exists() or not columns_path.exists():
        pytest.skip("Local model artifacts absent; run training for this check")
    model = joblib.load(model_path)
    columns = joblib.load(columns_path)
    assert columns == list(model.feature_names_in_)
    X = prepare_features(pd.DataFrame([valid_applicant]))[columns]
    probabilities = model.predict_proba(X)
    assert probabilities.shape == (1, 2)
    assert np.isfinite(probabilities).all()
    np.testing.assert_allclose(probabilities.sum(axis=1), 1)
