from uuid import UUID

import pytest


def test_prediction_log_matches_response_and_preserves_missing(
    client, valid_applicant, read_events,
):
    response = client.post("/predict", json={
        **valid_applicant, "EXT_SOURCE_3": None, "DAYS_EMPLOYED": 365243,
    })
    assert response.status_code == 200
    predictions = [event for event in read_events() if event["event"] == "prediction"]
    assert len(predictions) == 1
    event = predictions[0]
    assert event["request_id"] == response.headers["X-Request-ID"]
    UUID(event["request_id"])
    assert event["model_id"] == response.headers["X-Model-ID"]
    assert event["outputs"] == response.json()
    assert len(event["features"]) == 18
    assert event["features"]["EXT_SOURCE_3"] is None
    assert event["features"]["YEARS_EMPLOYED"] is None
    assert event["feature_representation"] == "engineered_features_before_imputation"
    assert event["inference_time_ms"] >= 0
    assert event["timestamp"]


def test_prediction_error_has_correlated_id_and_traceback(
    client, valid_applicant, monkeypatch, read_events,
):
    from app import main
    def fail(_):
        raise RuntimeError("Synthetic feature processing failure")
    monkeypatch.setattr(main, "prepare_features", fail)
    response = client.post("/predict", json=valid_applicant)
    assert response.status_code == 500
    request_id = response.headers["X-Request-ID"]
    assert request_id in response.json()["detail"]
    assert "Synthetic feature" not in response.json()["detail"]
    errors = [event for event in read_events() if event["event"] == "prediction_error"]
    assert len(errors) == 1
    assert errors[0]["request_id"] == request_id
    assert errors[0]["level"] == "ERROR"
    assert "Synthetic feature processing failure" in errors[0]["exception"]


def test_repeated_logger_access_does_not_duplicate_events(runtime, read_events):
    from monitoring.logger import get_logger, log_event
    get_logger()
    get_logger()
    log_event("test_event", value=123)
    events = read_events()
    assert len(events) == 1
    assert events[0]["value"] == 123


def test_client_context_runs_startup_and_shutdown(runtime, read_events):
    from fastapi.testclient import TestClient
    from app.main import app
    with TestClient(app) as client:
        assert client.get("/health").status_code == 200
        assert read_events()[0]["event"] == "startup"
    assert read_events()[-1]["event"] == "shutdown"
