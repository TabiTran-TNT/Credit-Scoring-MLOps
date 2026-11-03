from hashlib import sha256
from io import BytesIO
from pathlib import Path
import os
import json

import joblib

PROJECT_ROOT = Path(__file__).resolve().parents[1]
_model = None
_feature_columns = None
_model_id = None
_model_version = None


def _resolve_path(env_name, default):
    path = Path(os.environ.get(env_name, default))
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_model():
    """Load once and identify the exact model bytes that were loaded."""
    global _model, _feature_columns, _model_id, _model_version

    model_path = _resolve_path("MODEL_PATH", "model/credit_model.pkl")
    features_path = _resolve_path("FEATURES_PATH", "model/feature_columns.pkl")
    selected_version = None
    if os.environ.get("DEPLOYMENT_MANIFEST"):
        from credit_scoring.lifecycle import verify_release
        manifest_path = _resolve_path("DEPLOYMENT_MANIFEST", "deployments/current.json")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        paths = verify_release(manifest, manifest_path.parent)
        model_path, features_path = paths['model'], paths['columns']
        selected_version = manifest['version']
    model_bytes = model_path.read_bytes()
    model = joblib.load(BytesIO(model_bytes))
    columns = joblib.load(features_path)

    if not isinstance(columns, list) or not columns:
        raise ValueError("Feature columns must be a non-empty list.")
    if not all(isinstance(column, str) for column in columns):
        raise ValueError("Feature column names must be strings.")
    if len(columns) != len(set(columns)):
        raise ValueError("Duplicate feature columns found.")
    if not hasattr(model, "predict_proba"):
        raise ValueError("The saved model must support predict_proba().")
    if list(model.feature_names_in_) != columns:
        raise ValueError("Model and feature_columns.pkl do not match.")
    if list(model.classes_) != [0, 1]:
        raise ValueError("Expected class labels [0, 1].")

    _model = model
    _feature_columns = tuple(columns)
    _model_id = "sha256:" + sha256(model_bytes).hexdigest()
    _model_version = selected_version
    return _model


def get_model():
    if _model is None:
        raise RuntimeError("Model has not been loaded.")
    return _model


def get_feature_columns():
    if _feature_columns is None:
        raise RuntimeError("Feature columns have not been loaded.")
    return list(_feature_columns)


def get_model_id():
    if _model_id is None:
        raise RuntimeError("Model has not been loaded.")
    return _model_id


def get_model_version():
    return _model_version
