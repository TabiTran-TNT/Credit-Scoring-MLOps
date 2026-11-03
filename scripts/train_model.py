"""Train a candidate, save per-run outputs, and optionally track it in MLflow."""

import argparse
import json
from pathlib import Path
from uuid import uuid4

import joblib
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from credit_scoring.config import get_tracking_config
from credit_scoring.tracking import git_metadata, hash_file, log_training_model, training_run
from scripts.prepare_data import load_and_prepare_data

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "model"
RANDOM_STATE = 42


def main(n_estimators=200, max_depth=4, run_name=None):
    config = get_tracking_config()
    with training_run(config, run_name) as run:
        run_id = run.info.run_id if run else uuid4().hex
        dataset_path = DATA_DIR / "application_train.csv"
        dataset_hash = hash_file(dataset_path)
        X, y = load_and_prepare_data(file_path=str(dataset_path))
        if hash_file(dataset_path) != dataset_hash:
            raise RuntimeError("Dataset changed while it was being loaded")

        X_dev, X_test, y_dev, y_test = train_test_split(
            X, y, test_size=0.15, random_state=RANDOM_STATE, stratify=y,
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_dev, y_dev, test_size=0.15 / 0.85,
            random_state=RANDOM_STATE, stratify=y_dev,
        )
        print(f"Run ID: {run_id}")
        print(f"Train / validation / test: {len(X_train)} / {len(X_val)} / {len(X_test)}")
        pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("classifier", GradientBoostingClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=0.1,
                subsample=0.8,
                random_state=RANDOM_STATE,
            )),
        ])
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_val)
        probabilities = pipeline.predict_proba(X_val)[:, 1]
        metrics = {
            "validation_roc_auc": float(roc_auc_score(y_val, probabilities)),
            "validation_average_precision": float(average_precision_score(y_val, probabilities)),
        }
        print(classification_report(y_val, predictions, zero_division=0))
        print(json.dumps(metrics, indent=2))

        # Candidate outputs cannot overwrite the artifacts used by the API.
        model_dir = MODEL_DIR / "runs" / run_id
        data_dir = DATA_DIR / "runs" / run_id
        model_dir.mkdir(parents=True, exist_ok=False)
        data_dir.mkdir(parents=True, exist_ok=False)
        joblib.dump(pipeline, model_dir / "credit_model.pkl")
        joblib.dump(list(X_train.columns), model_dir / "feature_columns.pkl")
        X_train.to_csv(data_dir / "reference_data.csv", index=False)
        for name, features, labels in (
            ("validation", X_val, y_val), ("test", X_test, y_test),
        ):
            features.to_csv(data_dir / f"{name}_data.csv", index=False)
            labels.to_frame(name="TARGET").to_csv(data_dir / f"{name}_labels.csv", index=False)

        source_files = [
            PROJECT_ROOT / "scripts" / "train_model.py",
            PROJECT_ROOT / "scripts" / "prepare_data.py",
            PROJECT_ROOT / "credit_scoring" / "features.py",
            PROJECT_ROOT / "credit_scoring" / "config.py",
            PROJECT_ROOT / "credit_scoring" / "tracking.py",
        ]
        source_files = [path for path in source_files if path.is_file()]
        metadata = {
            "run_id": run_id,
            "mlflow_enabled": config.enabled,
            "tracking_uri": config.tracking_uri if config.enabled else None,
            "random_state": RANDOM_STATE,
            "train_rows": len(X_train),
            "validation_rows": len(X_val),
            "test_rows": len(X_test),
            "feature_columns": list(X_train.columns),
            **metrics,
            "dataset_sha256": dataset_hash,
            "dataset_filename": dataset_path.name,
            "reference_representation": "engineered_features_before_imputation",
            "model_parameters": pipeline.named_steps["classifier"].get_params(),
            "source_sha256": {str(path.relative_to(PROJECT_ROOT)): hash_file(path) for path in source_files},
            **git_metadata(PROJECT_ROOT),
        }
        if config.enabled:
            metadata.update(log_training_model(
                config, pipeline, X_train, metrics, metadata,
                model_dir, data_dir, source_files,
            ))

        metadata_path = model_dir / "training_metadata.json"
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        if config.enabled:
            import mlflow
            mlflow.log_artifact(str(metadata_path), artifact_path="metadata")

        result = {"run_id": run_id, "model_dir": str(model_dir), "data_dir": str(data_dir), **metadata}
        print(f"Candidate model directory: {model_dir}")
        print(f"Reference/evaluation data directory: {data_dir}")
        if config.enabled:
            print(f"Registered model: {metadata['registered_model_uri']}")
        return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-estimators", type=int, default=200)
    parser.add_argument("--max-depth", type=int, default=4)
    parser.add_argument("--run-name", default=None)
    args = parser.parse_args()
    main(args.n_estimators, args.max_depth, args.run_name)
