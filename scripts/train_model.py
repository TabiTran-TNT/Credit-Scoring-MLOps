from pathlib import Path
import json

import joblib
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from prepare_data import load_and_prepare_data


# Resolve paths relative to the project, regardless of working directory.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "model"

RANDOM_STATE = 42


def main():
    # ---------------------------------------------------------
    # 1. Load and prepare data
    # ---------------------------------------------------------
    # prepare_data.py performs fixed feature transformations.
    # Missing values must remain here; do not calculate medians yet.
    X, y = load_and_prepare_data(
        file_path=str(DATA_DIR / "application_train.csv")
    )

    print(f"Dataset: {X.shape[0]:,} rows, {X.shape[1]} features")
    print(f"Missing values before imputation: {X.isna().sum().sum():,}")

    # ---------------------------------------------------------
    # 2. Split: 70% train, 15% validation, 15% test
    # ---------------------------------------------------------
    X_dev, X_test, y_dev, y_test = train_test_split(
        X,
        y,
        test_size=0.15,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_dev,
        y_dev,
        test_size=0.15 / 0.85,
        random_state=RANDOM_STATE,
        stratify=y_dev,
    )

    print(f"\nTrain:      {len(X_train):,} rows")
    print(f"Validation: {len(X_val):,} rows")
    print(f"Test:       {len(X_test):,} rows")

    print(f"\nTrain default rate:      {y_train.mean():.2%}")
    print(f"Validation default rate: {y_val.mean():.2%}")
    print(f"Test default rate:       {y_test.mean():.2%}")

    # ---------------------------------------------------------
    # 3. Build the pipeline
    # ---------------------------------------------------------
    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("classifier", GradientBoostingClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.1,
            subsample=0.8,
            random_state=RANDOM_STATE,
        )),
    ])

    # ---------------------------------------------------------
    # 4. Train
    # ---------------------------------------------------------
    # The imputer learns column medians ONLY from X_train.
    # The classifier then trains on the imputed training data.
    print("\nTraining model...")
    pipeline.fit(X_train, y_train)

    # ---------------------------------------------------------
    # 5. Evaluate on validation data
    # ---------------------------------------------------------
    # The fitted imputer reuses the medians learned from training.
    y_pred = pipeline.predict(X_val)
    y_proba = pipeline.predict_proba(X_val)[:, 1]

    roc_auc = roc_auc_score(y_val, y_proba)
    average_precision = average_precision_score(y_val, y_proba)

    print("\nValidation classification report:")
    print(classification_report(y_val, y_pred, zero_division=0))
    print(f"Validation ROC-AUC: {roc_auc:.4f}")
    print(f"Validation average precision: {average_precision:.4f}")

    # Test data is reserved for final evaluation after model selection.

    # ---------------------------------------------------------
    # 6. Save the fitted pipeline and feature columns
    # ---------------------------------------------------------
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(pipeline, MODEL_DIR / "credit_model.pkl")
    joblib.dump(
        list(X_train.columns),
        MODEL_DIR / "feature_columns.pkl",
    )

    # ---------------------------------------------------------
    # 7. Save reference data and evaluation datasets
    # ---------------------------------------------------------
    # Reference data: engineered features BEFORE imputation.
    # Future drift comparisons must use the same representation.
    X_train.to_csv(DATA_DIR / "reference_data.csv", index=False)

    X_val.to_csv(DATA_DIR / "validation_data.csv", index=False)
    y_val.to_frame(name="TARGET").to_csv(
        DATA_DIR / "validation_labels.csv",
        index=False,
    )

    X_test.to_csv(DATA_DIR / "test_data.csv", index=False)
    y_test.to_frame(name="TARGET").to_csv(
        DATA_DIR / "test_labels.csv",
        index=False,
    )

    # ---------------------------------------------------------
    # 8. Save metrics and training information
    # ---------------------------------------------------------
    metadata = {
        "random_state": RANDOM_STATE,
        "train_rows": len(X_train),
        "validation_rows": len(X_val),
        "test_rows": len(X_test),
        "feature_columns": list(X_train.columns),
        "validation_roc_auc": float(roc_auc),
        "validation_average_precision": float(average_precision),
        "reference_representation": "engineered_features_before_imputation",
        "model_parameters": pipeline.named_steps["classifier"].get_params(),
    }

    with (MODEL_DIR / "training_metadata.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(metadata, file, indent=2)

    print(f"\nModel saved to: {MODEL_DIR / 'credit_model.pkl'}")
    print(f"Reference data saved to: {DATA_DIR / 'reference_data.csv'}")
    print("Validation/test features and labels saved.")
    print("Training completed.")


if __name__ == "__main__":
    main()
