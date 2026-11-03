from hashlib import sha256
import json
from pathlib import Path

import joblib
import mlflow
import mlflow.pyfunc
import mlflow.sklearn
import numpy as np
import pandas as pd
import pytest

from credit_scoring.config import get_tracking_config
from credit_scoring.features import prepare_features
from credit_scoring.tracking import hash_file, training_run


@pytest.fixture
def tracking_environment(tmp_path, monkeypatch, synthetic_data):
    from scripts import train_model
    previous_tracking = mlflow.get_tracking_uri()
    previous_registry = mlflow.get_registry_uri()
    data_dir = tmp_path / "data"
    model_dir = tmp_path / "model"
    data_dir.mkdir()
    model_dir.mkdir()
    raw, y = synthetic_data
    raw.assign(TARGET=y).to_csv(data_dir / "application_train.csv", index=False)
    monkeypatch.setattr(train_model, "DATA_DIR", data_dir)
    monkeypatch.setattr(train_model, "MODEL_DIR", model_dir)
    uri = "sqlite:///" + (tmp_path / "tracking.db").as_posix()
    monkeypatch.setenv("MLFLOW_ENABLED", "true")
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    monkeypatch.setenv("MLFLOW_EXPERIMENT_NAME", "test-credit-scoring")
    monkeypatch.setenv("MLFLOW_REGISTERED_MODEL_NAME", "test-credit-scoring")
    mlflow.set_tracking_uri(uri)
    mlflow.set_registry_uri(uri)
    client = mlflow.MlflowClient()
    client.create_experiment(
        "test-credit-scoring", artifact_location=(tmp_path / "artifacts").as_uri(),
    )
    try:
        yield train_model, client, data_dir, model_dir
    finally:
        mlflow.set_tracking_uri(previous_tracking)
        mlflow.set_registry_uri(previous_registry)


def test_config_respects_explicit_environment(monkeypatch):
    monkeypatch.setenv("MLFLOW_ENABLED", "false")
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5555")
    config = get_tracking_config()
    assert config.enabled is False
    assert config.tracking_uri == "http://127.0.0.1:5555"


def test_config_rejects_invalid_flag(monkeypatch):
    monkeypatch.setenv("MLFLOW_ENABLED", "maybe")
    with pytest.raises(ValueError, match="MLFLOW_ENABLED"):
        get_tracking_config()


def test_hash_detects_changed_data(tmp_path):
    path = tmp_path / "data.csv"
    path.write_bytes(b"a,b\n1,2\n")
    before = hash_file(path)
    assert before == sha256(path.read_bytes()).hexdigest()
    path.write_bytes(b"a,b\n1,3\n")
    assert hash_file(path) != before


def test_tracking_registry_roundtrip_and_candidate_isolation(tracking_environment):
    train_model, client, data_dir, model_dir = tracking_environment
    # Protect current serving artifacts, including an old reference dataset.
    serving_path = model_dir / "credit_model.pkl"
    serving_path.write_bytes(b"existing-serving-model-must-stay-unchanged")
    reference_path = data_dir / "reference_data.csv"
    reference_path.write_text("existing reference", encoding="utf-8")
    first = train_model.main(n_estimators=8, max_depth=2, run_name="baseline")
    second = train_model.main(n_estimators=12, max_depth=3, run_name="comparison")
    assert serving_path.read_bytes() == b"existing-serving-model-must-stay-unchanged"
    assert reference_path.read_text() == "existing reference"
    assert first["run_id"] != second["run_id"]
    assert first["registered_model_version"] == "1"
    assert second["registered_model_version"] == "2"
    assert mlflow.active_run() is None

    for result, estimators in ((first, 8), (second, 12)):
        run = client.get_run(result["run_id"])
        assert run.info.status == "FINISHED"
        assert run.data.params["n_estimators"] == str(estimators)
        assert run.data.metrics["validation_roc_auc"] == result["validation_roc_auc"]
        assert run.data.tags["dataset_sha256"] == hash_file(data_dir / "application_train.csv")
        version = client.get_model_version("test-credit-scoring", result["registered_model_version"])
        assert version.tags["validation_status"] == "pending"
        assert version.run_id == result["run_id"]
        assert client.get_registered_model("test-credit-scoring").aliases == {}

        local = joblib.load(Path(result["model_dir"]) / "credit_model.pkl")
        restored = mlflow.sklearn.load_model(result["registered_model_uri"])
        validation = pd.read_csv(Path(result["data_dir"]) / "validation_data.csv")
        np.testing.assert_allclose(restored.predict_proba(validation), local.predict_proba(validation))
        generic = mlflow.pyfunc.load_model(result["registered_model_uri"])
        np.testing.assert_allclose(generic.predict(validation.astype(float)), local.predict_proba(validation))
        reference = client.download_artifacts(result["run_id"], "reference/reference_data.csv")
        assert pd.read_csv(reference).isna().any().any()
        metadata_file = client.download_artifacts(result["run_id"], "metadata/training_metadata.json")
        metadata = json.loads(Path(metadata_file).read_text())
        assert metadata["registered_model_version"] == result["registered_model_version"]
        assert metadata["source_sha256"]
        assert "test_roc_auc" not in run.data.metrics


def test_failure_marks_run_failed_and_closes_context(tracking_environment):
    _, client, _, _ = tracking_environment
    config = get_tracking_config()
    with pytest.raises(RuntimeError, match="Synthetic"):
        with training_run(config, "failed-run") as run:
            run_id = run.info.run_id
            raise RuntimeError("Synthetic training failure")
    assert client.get_run(run_id).info.status == "FAILED"
    assert mlflow.active_run() is None
