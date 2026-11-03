"""Evaluate registry candidates on reproducible, common validation data."""
from datetime import datetime, timezone
import json
from pathlib import Path
from time import perf_counter
from uuid import uuid4
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field
from sklearn.metrics import average_precision_score, roc_auc_score
import yaml
from credit_scoring.artifacts import ROOT, atomic_json, canonical_hash, download, registry_client, run_artifact, tree_hash
from credit_scoring.tracking import hash_file


class ValidationRules(BaseModel):
    model_config = ConfigDict(extra='forbid', allow_inf_nan=False)
    min_rows: int = Field(default=100, ge=2)
    min_roc_auc: float = Field(default=.70, ge=0, le=1)
    min_average_precision: float = Field(default=.10, ge=0, le=1)
    max_auc_drop: float = Field(default=.005, ge=0, le=1)
    max_p95_latency_ms: float = Field(default=100, gt=0)
    latency_repeats: int = Field(default=30, ge=5, le=1000)
    max_report_age_hours: float = Field(default=24, gt=0)


def load_rules(path):
    return ValidationRules.model_validate(yaml.safe_load(Path(path).read_text(encoding='utf-8')))


def champion_version(client, name):
    aliases = client.get_registered_model(name).aliases
    return str(aliases['champion']) if 'champion' in aliases else None


def _contract(metadata):
    contract = metadata.get('evaluation_contract')
    if not contract:
        raise ValueError('Model lacks an evaluation contract; retrain using the current code')
    return contract


def validate_model(name, version, rules, baseline=None):
    client = registry_client()
    version = str(version)
    candidate = client.get_model_version(name, version)
    client.set_model_version_tag(name, version, 'validation_status', 'running')
    try:
        active = champion_version(client, name)
        if baseline is not None and str(baseline) != active:
            raise ValueError('Explicit baseline must be the current champion')
        metadata = json.loads(run_artifact(client, candidate.run_id, 'metadata/training_metadata.json').read_text())
        contract = _contract(metadata)
        features_file = run_artifact(client, candidate.run_id, 'evaluation/validation_data.csv')
        labels_file = run_artifact(client, candidate.run_id, 'evaluation/validation_labels.csv')
        if hash_file(features_file) != contract['features_sha256'] or hash_file(labels_file) != contract['labels_sha256']:
            raise ValueError('Evaluation data does not match its recorded fingerprint')
        X = pd.read_csv(features_file)
        labels = pd.read_csv(labels_file)
        if list(labels.columns) != ['TARGET'] or len(X) != len(labels):
            raise ValueError('Feature/label schema or row count mismatch')
        y = labels['TARGET']
        if set(y.unique()) != {0, 1} or y.isna().any():
            raise ValueError('Evaluation labels must contain both binary classes')
        artifact = download(f'models:/{name}/{version}')
        model = mlflow.sklearn.load_model(str(artifact))
        if list(X.columns) != list(model.feature_names_in_) or 'TARGET' in X:
            raise ValueError('Evaluation feature order does not match the model')
        if not all(pd.api.types.is_numeric_dtype(t) for t in X.dtypes) or np.isinf(X.to_numpy()).any():
            raise ValueError('Non-numeric or infinite features')
        if list(model.classes_) != [0, 1]:
            raise ValueError('Unexpected model classes')
        probabilities = model.predict_proba(X)[:, 1]
        if not np.isfinite(probabilities).all() or not ((probabilities >= 0) & (probabilities <= 1)).all():
            raise ValueError('Invalid probabilities')
        metrics = {'roc_auc': float(roc_auc_score(y, probabilities)),
                   'average_precision': float(average_precision_score(y, probabilities))}
        model.predict_proba(X.iloc[:1])
        times = []
        for _ in range(rules.latency_repeats):
            started = perf_counter()
            model.predict_proba(X.iloc[:1])
            times.append((perf_counter() - started) * 1000)
        metrics['p95_latency_ms'] = float(np.percentile(times, 95))
        checks = [
            {'name': 'minimum_rows', 'passed': len(X) >= rules.min_rows},
            {'name': 'roc_auc', 'passed': metrics['roc_auc'] >= rules.min_roc_auc},
            {'name': 'average_precision', 'passed': metrics['average_precision'] >= rules.min_average_precision},
            {'name': 'latency', 'passed': metrics['p95_latency_ms'] <= rules.max_p95_latency_ms},
        ]
        if active is not None and active != version:
            old = client.get_model_version(name, active)
            old_meta = json.loads(run_artifact(client, old.run_id, 'metadata/training_metadata.json').read_text())
            if _contract(old_meta) != contract:
                raise ValueError('Champion and candidate do not share the same evaluation contract')
            old_model = mlflow.sklearn.load_model(str(download(f'models:/{name}/{active}')))
            metrics['baseline_roc_auc'] = float(roc_auc_score(y, old_model.predict_proba(X)[:, 1]))
            checks.append({'name': 'champion_regression', 'passed':
                           metrics['roc_auc'] >= metrics['baseline_roc_auc'] - rules.max_auc_drop})
        report = {
            'schema_version': 1, 'validation_id': uuid4().hex,
            'model_name': name, 'version': version, 'baseline_version': active,
            'created_at': datetime.now(timezone.utc).isoformat(),
            'model_sha256': tree_hash(artifact), 'evaluation_contract': contract,
            'rules': rules.model_dump(), 'metrics': metrics, 'checks': checks,
            'passed': all(c['passed'] for c in checks),
        }
        with mlflow.start_run(experiment_id=client.get_run(candidate.run_id).info.experiment_id,
                              run_name=f'validate-{name}-v{version}') as run:
            mlflow.log_dict(report, 'validation/report.json')
            mlflow.log_metrics(metrics)
            validation_run_id = run.info.run_id
        for key, value in {
            'validation_status': 'passed' if report['passed'] else 'failed',
            'validation_run_id': validation_run_id,
            'validation_report_sha256': canonical_hash(report),
        }.items():
            client.set_model_version_tag(name, version, key, value)
        atomic_json(ROOT / 'reports' / 'validation' / (report['validation_id'] + '.json'), report)
        return report
    except Exception:
        client.set_model_version_tag(name, version, 'validation_status', 'failed')
        raise
