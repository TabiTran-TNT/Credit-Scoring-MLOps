import numpy as np
import pandas as pd
import pytest

from credit_scoring.features import prepare_features


def test_health_reports_loaded_model(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy", "model_loaded": True}


def test_prediction_matches_direct_pipeline(client, valid_applicant):
    from app.model_loader import get_model, get_feature_columns
    response = client.post("/predict", json=valid_applicant)
    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"prediction", "probability_of_default", "risk_category"}
    assert body["prediction"] in (0, 1)
    assert 0 <= body["probability_of_default"] <= 1
    assert body["risk_category"] in ("Low", "Medium", "High")
    X = prepare_features(pd.DataFrame([valid_applicant]))[get_feature_columns()]
    model = get_model()
    assert body["probability_of_default"] == pytest.approx(
        float(model.predict_proba(X)[0, 1]), abs=1e-12,
    )
    assert body["prediction"] == int(model.predict(X)[0])
    assert response.headers["X-Request-ID"]
    assert response.headers["X-Model-ID"].startswith("sha256:")


@pytest.mark.parametrize("field", [
    "EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3", "DAYS_EMPLOYED",
    "AMT_ANNUITY", "AMT_GOODS_PRICE",
])
def test_nullable_fields_are_scored(client, valid_applicant, field):
    response = client.post("/predict", json={**valid_applicant, field: None})
    assert response.status_code == 200, response.text


@pytest.mark.parametrize("field,value", [
    ("EXT_SOURCE_1", 5),
    ("EXT_SOURCE_2", -0.1),
    ("AMT_INCOME_TOTAL", -1000),
    ("AMT_CREDIT", 0),
    ("DAYS_BIRTH", "forty"),
    ("DAYS_BIRTH", -15 * 365),
    ("DAYS_EMPLOYED", 10),
    ("CODE_GENDER", "unknown"),
])
def test_invalid_fields_return_422(client, valid_applicant, field, value):
    response = client.post("/predict", json={**valid_applicant, field: value})
    assert response.status_code == 422
    assert any(field in error["loc"] for error in response.json()["detail"])


def test_missing_required_field_is_rejected(client, valid_applicant):
    valid_applicant.pop("EXT_SOURCE_2")
    assert client.post("/predict", json=valid_applicant).status_code == 422


def test_extra_target_is_rejected(client, valid_applicant):
    response = client.post("/predict", json={**valid_applicant, "TARGET": 1})
    assert response.status_code == 422


@pytest.mark.parametrize("probability,category,prediction", [
    (0.2999, "Low", 0),
    (0.3, "Medium", 0),
    (0.5, "Medium", 1),
    (0.5999, "Medium", 1),
    (0.6, "High", 1),
])
def test_score_threshold_boundaries(
    client, valid_applicant, monkeypatch, probability, category, prediction,
):
    from app import main
    class ConstantModel:
        def predict_proba(self, X):
            return np.array([[1 - probability, probability]])
    monkeypatch.setattr(main, "get_model", lambda: ConstantModel())
    body = client.post("/predict", json=valid_applicant).json()
    assert body["risk_category"] == category
    assert body["prediction"] == prediction


def test_unavailable_model_returns_unhealthy(client, monkeypatch):
    from app import main
    def unavailable():
        raise RuntimeError("Model unavailable")
    monkeypatch.setattr(main, "get_model", unavailable)
    response = client.get("/health")
    assert response.status_code == 503
    assert response.json()["model_loaded"] is False
