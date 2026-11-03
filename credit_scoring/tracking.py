"""Explicit MLflow logging; promotion and serving are separate later steps."""

from contextlib import contextmanager
from hashlib import sha256
from importlib.metadata import version
from pathlib import Path
import os
import subprocess


def hash_file(path):
    digest = sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_metadata(project_root):
    def git(*args):
        try:
            result = subprocess.run(
                ["git", "-C", str(project_root), *args],
                capture_output=True, text=True, timeout=10,
            )
            return result.stdout.strip() if result.returncode == 0 else None
        except (OSError, subprocess.TimeoutExpired):
            return None
    commit = git("rev-parse", "HEAD")
    status = git("status", "--porcelain") if commit else None
    return {"git_commit": commit, "git_dirty": None if status is None else bool(status)}


@contextmanager
def training_run(config, run_name=None):
    if not config.enabled:
        yield None
        return

    # MLflow's URL banners contain emoji that fail with some Windows encodings.
    # The CLI already prints the run ID and registered model URI in plain text.
    os.environ.setdefault("MLFLOW_SUPPRESS_PRINTING_URL_TO_STDOUT", "true")
    import mlflow
    mlflow.set_tracking_uri(config.tracking_uri)
    mlflow.set_registry_uri(config.tracking_uri)
    mlflow.set_experiment(config.experiment_name)
    # A connection error here occurs before expensive training starts.
    with mlflow.start_run(run_name=run_name) as run:
        yield run


def log_training_model(config, pipeline, X_train, metrics, metadata, model_dir, data_dir, source_files):
    import mlflow
    import mlflow.sklearn
    from mlflow.models import infer_signature

    mlflow.log_params({
        **metadata["model_parameters"],
        "split_seed": metadata["random_state"],
        "split_train": 0.70,
        "split_validation": 0.15,
        "split_test": 0.15,
        "imputer_strategy": "median",
        "n_features": len(X_train.columns),
    })
    mlflow.log_metrics(metrics)
    mlflow.set_tags({
        "dataset_sha256": metadata["dataset_sha256"],
        "git_commit": metadata["git_commit"] or "unavailable",
        "git_dirty": str(metadata["git_dirty"]),
        "feature_representation": metadata["reference_representation"],
        "input_representation": "engineered_features_not_raw_request",
    })

    # Float schema supports missing numeric inputs without integer/null issues.
    example = X_train.head(5).astype(float)
    signature = infer_signature(example, pipeline.predict_proba(example))
    requirements = [
        f"{name}=={version(name)}" for name in
        ("scikit-learn", "numpy", "pandas", "scipy", "joblib", "cloudpickle")
    ]
    model_info = mlflow.sklearn.log_model(
        sk_model=pipeline,
        name="pipeline",
        serialization_format="cloudpickle",
        signature=signature,
        input_example=example,
        pyfunc_predict_fn="predict_proba",
        pip_requirements=requirements,
        metadata={"input_representation": "engineered_features_before_imputation"},
    )
    model_version = mlflow.register_model(
        model_uri=model_info.model_uri,
        name=config.registered_model_name,
    )
    client = mlflow.MlflowClient()
    client.set_model_version_tag(
        config.registered_model_name, model_version.version,
        "validation_status", "pending",
    )
    mlflow.log_artifact(str(model_dir / "feature_columns.pkl"), artifact_path="metadata")
    mlflow.log_artifact(str(data_dir / "reference_data.csv"), artifact_path="reference")
    for path in source_files:
        # Keep original paths; features.py and train_model.py must not collide.
        artifact_dir = "source/" + path.parent.name
        mlflow.log_artifact(str(path), artifact_path=artifact_dir)
    return {
        "model_uri": model_info.model_uri,
        "registered_model_name": config.registered_model_name,
        "registered_model_version": str(model_version.version),
        "registered_model_uri": f"models:/{config.registered_model_name}/{model_version.version}",
    }
