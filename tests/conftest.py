"""Small synthetic fixtures; tests never train on the full credit dataset."""

import json
import logging
import os
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

PROJECT_ROOT = Path(os.environ.get(
    "TEST_PROJECT_ROOT", str(Path(__file__).resolve().parents[1])
)).resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from credit_scoring.features import prepare_features


@pytest.fixture
def valid_applicant():
    return {
        "EXT_SOURCE_1": 0.5,
        "EXT_SOURCE_2": 0.65,
        "EXT_SOURCE_3": 0.48,
        "DAYS_BIRTH": -14610,
        "DAYS_EMPLOYED": -1826,
        "DAYS_ID_PUBLISH": -3650,
        "AMT_INCOME_TOTAL": 202500,
        "AMT_CREDIT": 406597,
        "AMT_ANNUITY": 24700,
        "AMT_GOODS_PRICE": 351000,
        "CODE_GENDER": "F",
        "FLAG_OWN_CAR": "N",
        "FLAG_OWN_REALTY": "Y",
        "CNT_CHILDREN": 0,
        "NAME_EDUCATION_TYPE": "Higher education",
    }


@pytest.fixture(scope="session")
def synthetic_data():
    rng = np.random.default_rng(42)
    n = 120
    raw = pd.DataFrame({
        "EXT_SOURCE_1": rng.uniform(0, 1, n),
        "EXT_SOURCE_2": rng.uniform(0, 1, n),
        "EXT_SOURCE_3": rng.uniform(0, 1, n),
        "DAYS_BIRTH": rng.integers(-25000, -8000, n),
        "DAYS_EMPLOYED": rng.integers(-6000, 0, n),
        "DAYS_ID_PUBLISH": rng.integers(-5000, 0, n),
        "AMT_INCOME_TOTAL": rng.uniform(50000, 300000, n),
        # Unique values let the training test identify split membership.
        "AMT_CREDIT": 100000 + np.arange(n) * 1000,
        "AMT_ANNUITY": rng.uniform(10000, 40000, n),
        "AMT_GOODS_PRICE": rng.uniform(80000, 300000, n),
        "CODE_GENDER": np.where(np.arange(n) % 2, "F", "M"),
        "FLAG_OWN_CAR": "N",
        "FLAG_OWN_REALTY": "Y",
        "CNT_CHILDREN": np.arange(n) % 3,
        "NAME_EDUCATION_TYPE": "Higher education",
    })
    raw.loc[::7, "EXT_SOURCE_3"] = np.nan
    raw.loc[::9, "DAYS_EMPLOYED"] = 365243
    y = pd.Series(np.arange(n) % 2, name="TARGET")
    return raw, y


@pytest.fixture(scope="session")
def model_artifacts(tmp_path_factory, synthetic_data):
    directory = tmp_path_factory.mktemp("synthetic-model")
    raw, y = synthetic_data
    X = prepare_features(raw)
    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("classifier", GradientBoostingClassifier(
            n_estimators=8, max_depth=2, random_state=42,
        )),
    ])
    model.fit(X, y)
    model_path = directory / "credit_model.pkl"
    columns_path = directory / "feature_columns.pkl"
    joblib.dump(model, model_path)
    joblib.dump(list(X.columns), columns_path)
    return {"model": model_path, "columns": columns_path}


@pytest.fixture
def runtime(monkeypatch, tmp_path, model_artifacts):
    """Redirect artifacts/logs and isolate module globals for each test."""
    from app import model_loader
    from monitoring.logger import LOGGER_NAME

    monkeypatch.setenv("MODEL_PATH", str(model_artifacts["model"]))
    monkeypatch.setenv("FEATURES_PATH", str(model_artifacts["columns"]))
    monkeypatch.setenv("LOG_DIR", str(tmp_path / "logs"))
    monkeypatch.setenv("DEPLOYMENT_MANIFEST", "")
    for name in ("_model", "_feature_columns", "_model_id", "_model_version"):
        monkeypatch.setattr(model_loader, name, None)

    logger = logging.getLogger(LOGGER_NAME)
    previous = (
        list(logger.handlers), logger.level, logger.propagate, logger.disabled,
        getattr(logger, "_credit_scoring_configured", None),
    )
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
    if hasattr(logger, "_credit_scoring_configured"):
        delattr(logger, "_credit_scoring_configured")
    logger.disabled = False

    try:
        yield {**model_artifacts, "log_file": tmp_path / "logs/predictions.jsonl"}
    finally:
        for handler in list(logger.handlers):
            logger.removeHandler(handler)
            handler.close()
        handlers, level, propagate, disabled, configured = previous
        for handler in handlers:
            logger.addHandler(handler)
        logger.setLevel(level)
        logger.propagate = propagate
        logger.disabled = disabled
        if configured is None:
            if hasattr(logger, "_credit_scoring_configured"):
                delattr(logger, "_credit_scoring_configured")
        else:
            logger._credit_scoring_configured = configured


@pytest.fixture
def client(runtime):
    from fastapi.testclient import TestClient
    from app.main import app

    # The context manager runs startup and shutdown, including load_model().
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def read_events(runtime):
    def read():
        def reject_constant(value):
            raise AssertionError(f"Nonstandard JSON constant in log: {value}")
        return [
            json.loads(line, parse_constant=reject_constant)
            for line in runtime["log_file"].read_text(encoding="utf-8").splitlines()
        ]
    return read
