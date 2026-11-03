"""Windowed feature/score monitoring; drift triggers investigation, not retraining."""
from collections import deque
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from credit_scoring.artifacts import atomic_json

CATEGORICAL = {'CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY', 'CNT_CHILDREN', 'EDUCATION_LEVEL'}


def psi(reference, current, categorical=False):
    """Reference-defined bins, separate missing bin and additive smoothing."""
    reference, current = pd.Series(reference), pd.Series(current)
    if reference.empty or current.empty:
        raise ValueError('Both PSI populations must contain rows')
    if categorical:
        keys = sorted(set(reference.dropna().astype(str)) | set(current.dropna().astype(str)))
        counts = [[int((s.dropna().astype(str) == key).sum()) for key in keys] + [int(s.isna().sum())]
                  for s in (reference, current)]
    else:
        finite = reference.dropna().to_numpy(dtype=float)
        if np.isinf(finite).any() or np.isinf(current.dropna().to_numpy(dtype=float)).any():
            raise ValueError('PSI input contains infinity')
        internal = np.unique(np.quantile(finite, np.linspace(0, 1, 11)))[1:-1] if len(finite) else []
        if len(finite) and np.ptp(finite) == 0:
            epsilon = max(abs(float(finite[0])) * 1e-6, 1e-6)
            internal = [finite[0] - epsilon, finite[0] + epsilon]
        edges = np.r_[-np.inf, internal, np.inf]
        counts = [list(np.histogram(s.dropna(), bins=edges)[0]) + [int(s.isna().sum())]
                  for s in (reference, current)]
    proportions = [(np.asarray(c, dtype=float) + 1e-6) / (sum(c) + 1e-6 * len(c)) for c in counts]
    p, q = proportions
    return float(np.sum((q - p) * np.log(q / p)))


def log_window(path, model_id, since=None, max_rows=20000):
    if max_rows <= 0:
        raise ValueError('max_rows must be positive')
    records, malformed = deque(maxlen=max_rows), 0
    cutoff = pd.Timestamp(since) if since else None
    if cutoff is not None and cutoff.tzinfo is None:
        raise ValueError('--since must include a timezone')
    with Path(path).open(encoding='utf-8') as stream:
        for line in stream:
            if not line.strip():
                continue
            try:
                item = json.loads(line)
                timestamp = pd.Timestamp(item['timestamp'])
                if timestamp.tzinfo is None:
                    raise ValueError('Log timestamp lacks timezone')
                if cutoff is not None and timestamp < cutoff:
                    continue
                if item.get('model_id') != model_id:
                    continue
                if item.get('event') == 'prediction':
                    if item.get('feature_representation') != 'engineered_features_before_imputation':
                        raise ValueError('Unknown feature representation')
                    probability = float(item['outputs']['probability_of_default'])
                    if not np.isfinite(probability) or not 0 <= probability <= 1:
                        raise ValueError('Invalid score')
                    records.append(item)
                elif item.get('event') == 'prediction_error':
                    records.append(item)
            except (ValueError, KeyError, TypeError):
                malformed += 1
    window = list(records)
    return [r for r in window if r['event'] == 'prediction'], sum(r['event'] == 'prediction_error' for r in window), malformed


def matured_performance(records, label_path, as_of=None):
    """Only labels explicitly marked mature and available by the evaluation time."""
    labels = pd.read_csv(label_path, dtype={'request_id': str})
    required = {'request_id', 'TARGET', 'label_available_at', 'is_mature'}
    if not required.issubset(labels) or labels['request_id'].duplicated().any():
        raise ValueError('Labels require unique request_id, TARGET, label_available_at and is_mature')
    if not labels['is_mature'].isin([True, False, 0, 1]).all():
        raise ValueError('is_mature must be boolean or 0/1')
    now = pd.Timestamp(as_of or datetime.now(timezone.utc))
    if now.tzinfo is None:
        raise ValueError('as_of must include a timezone')
    available = pd.to_datetime(labels['label_available_at'], utc=True, errors='raise')
    labels = labels.loc[labels['is_mature'].astype(bool) & (available <= now)]
    predictions = pd.DataFrame([{'request_id': r['request_id'],
                                 'score': r['outputs']['probability_of_default']} for r in records])
    if predictions.empty:
        return {'status': 'insufficient_mature_labels', 'rows': 0}
    if predictions['request_id'].duplicated().any():
        raise ValueError('Duplicate prediction request IDs')
    joined = predictions.merge(labels, on='request_id', validate='one_to_one')
    if not joined['TARGET'].isin([0, 1]).all():
        raise ValueError('Mature labels must be binary')
    if len(joined) < 20 or joined['TARGET'].nunique() != 2:
        return {'status': 'insufficient_mature_labels', 'rows': len(joined)}
    return {'status': 'evaluated', 'rows': len(joined),
            'roc_auc': float(roc_auc_score(joined['TARGET'], joined['score'])),
            'average_precision': float(average_precision_score(joined['TARGET'], joined['score']))}


def render_evidently(reference, current, output_dir):
    from evidently import Report, Dataset, DataDefinition
    from evidently.presets import DataDriftPreset
    categorical = [c for c in reference if c in CATEGORICAL]
    definition = DataDefinition(numerical_columns=[c for c in reference if c not in categorical],
                                categorical_columns=categorical)
    baseline = Dataset.from_pandas(reference, data_definition=definition)
    production = Dataset.from_pandas(current, data_definition=definition)
    snapshot = Report([DataDriftPreset()]).run(current_data=production, reference_data=baseline)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    snapshot.save_html(str(output / 'evidently.html'))
    snapshot.save_json(str(output / 'evidently.json'))


def build_report(reference, records, model, model_id, output_dir, *, errors=0, malformed=0,
                 psi_threshold=.2, min_rows=100, labels=None, render=True):
    if not 0 < psi_threshold or min_rows < 2:
        raise ValueError('Invalid monitoring thresholds')
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if not records or reference.empty:
        raise ValueError('No successful predictions for the selected model/window')
    current = pd.DataFrame([r['features'] for r in records])
    if set(current.columns) != set(reference.columns):
        raise ValueError('Reference and production feature schemas differ')
    current = current.loc[:, reference.columns]
    if list(reference.columns) != list(model.feature_names_in_):
        raise ValueError('Reference does not match the model feature order')
    per_feature = {
        column: {'psi': psi(reference[column], current[column], column in CATEGORICAL),
                 'reference_missing_rate': float(reference[column].isna().mean()),
                 'current_missing_rate': float(current[column].isna().mean())}
        for column in reference
    }
    reference_scores = model.predict_proba(reference)[:, 1]
    scores = [r['outputs']['probability_of_default'] for r in records]
    score_psi = psi(reference_scores, scores)
    alerts = []
    if len(current) < min_rows:
        alerts.append('insufficient_rows')
    else:
        if any(v['psi'] >= psi_threshold for v in per_feature.values()):
            alerts.append('feature_drift')
        if score_psi >= psi_threshold:
            alerts.append('score_drift')
    if any(abs(v['current_missing_rate'] - v['reference_missing_rate']) > .1 for v in per_feature.values()):
        alerts.append('missing_rate_shift')
    if malformed:
        alerts.append('malformed_log_records')
    if errors / max(errors + len(records), 1) > .01:
        alerts.append('prediction_errors')
    performance = matured_performance(records, labels) if labels else {'status': 'labels_not_supplied'}
    if performance.get('status') == 'evaluated' and performance['roc_auc'] < .7:
        alerts.append('mature_label_performance')
    report = {
        'timestamp': datetime.now(timezone.utc).isoformat(), 'model_id': model_id,
        'rows': len(current), 'reference_rows': len(reference), 'score_psi': score_psi,
        'features': per_feature, 'performance': performance,
        'pipeline_health': {'prediction_errors': errors, 'malformed_records': malformed,
                            'error_rate': errors / max(errors + len(records), 1),
                            'p95_handler_latency_ms': float(np.percentile([r['inference_time_ms'] for r in records], 95))},
        'alerts': alerts, 'action_required': bool(alerts),
        'action': 'Investigate data/pipeline changes; evaluate mature labels before retraining or rollback.',
    }
    atomic_json(output / 'summary.json', report)
    if alerts:
        with (output / 'alerts.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(report, allow_nan=False) + '\n')
    if render:
        render_evidently(reference, current, output)
    return report
