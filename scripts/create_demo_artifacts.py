"""Create a small SYNTHETIC model for CI/container smoke checks only."""
import argparse
from pathlib import Path
import joblib
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from credit_scoring.artifacts import atomic_json
from credit_scoring.demo import synthetic_applications
from credit_scoring.features import prepare_features


def create_demo_artifacts(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    raw = synthetic_applications()
    X = prepare_features(raw)
    model = Pipeline([('imputer', SimpleImputer(strategy='median')),
                      ('classifier', GradientBoostingClassifier(n_estimators=8, max_depth=2, random_state=42))])
    model.fit(X, raw['TARGET'])
    joblib.dump(model, output / 'credit_model.pkl')
    joblib.dump(list(X.columns), output / 'feature_columns.pkl')
    X.to_csv(output / 'reference_data.csv', index=False)
    atomic_json(output / 'metadata.json', {'synthetic': True, 'purpose': 'CI smoke checks only'})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='work/demo-model')
    create_demo_artifacts(parser.parse_args().output)
