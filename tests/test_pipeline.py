"""Exercise the real training entry point on 120 synthetic rows, offline."""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from credit_scoring.features import prepare_features


def test_training_splits_before_imputation_and_saves_artifacts(
    monkeypatch, tmp_path, synthetic_data,
):
    from scripts import train_model
    raw, y = synthetic_data
    X = prepare_features(raw)
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    raw.assign(TARGET=y).to_csv(data_dir / "application_train.csv", index=False)
    fitted_rows = []
    original_fit = Pipeline.fit

    def record_fit(self, features, labels, **kwargs):
        fitted_rows.append(features.index.copy())
        return original_fit(self, features, labels, **kwargs)

    monkeypatch.setattr(Pipeline, "fit", record_fit)
    monkeypatch.setattr(train_model, "DATA_DIR", data_dir)
    monkeypatch.setattr(train_model, "MODEL_DIR", tmp_path / "model")
    monkeypatch.setenv("MLFLOW_ENABLED", "false")
    result = train_model.main()

    assert len(fitted_rows) == 1
    train_X = X.loc[fitted_rows[0]]
    model_dir, data_dir = Path(result["model_dir"]), Path(result["data_dir"])
    model = joblib.load(model_dir / "credit_model.pkl")
    imputer = model.named_steps["imputer"]
    np.testing.assert_allclose(imputer.statistics_, train_X.median().to_numpy())
    assert not np.allclose(train_X.median(), X.median())

    reference = pd.read_csv(data_dir / "reference_data.csv")
    validation = pd.read_csv(data_dir / "validation_data.csv")
    test = pd.read_csv(data_dir / "test_data.csv")
    pd.testing.assert_frame_equal(reference, train_X.reset_index(drop=True), check_dtype=False)
    assert reference.isna().any().any()
    memberships = [set(frame["AMT_CREDIT"]) for frame in (reference, validation, test)]
    assert memberships[0].isdisjoint(memberships[1])
    assert memberships[0].isdisjoint(memberships[2])
    assert memberships[1].isdisjoint(memberships[2])
    assert set.union(*memberships) == set(X["AMT_CREDIT"])
    assert len(pd.read_csv(data_dir / "validation_labels.csv")) == len(validation)
    assert len(pd.read_csv(data_dir / "test_labels.csv")) == len(test)
    assert joblib.load(model_dir / "feature_columns.pkl") == list(X.columns)

    before = imputer.statistics_.copy()
    shifted = validation.copy()
    shifted["AMT_INCOME_TOTAL"] *= 100
    shifted["EXT_SOURCE_3"] = np.nan
    assert np.isfinite(model.predict_proba(shifted)).all()
    np.testing.assert_array_equal(imputer.statistics_, before)
    transformed = imputer.transform(shifted)
    np.testing.assert_allclose(
        transformed[:, list(X.columns).index("EXT_SOURCE_3")],
        before[list(X.columns).index("EXT_SOURCE_3")],
    )
    metadata = json.loads((model_dir / "training_metadata.json").read_text())
    assert metadata["train_rows"] == len(reference)
    assert metadata["mlflow_enabled"] is False
    assert metadata["reference_representation"] == "engineered_features_before_imputation"
