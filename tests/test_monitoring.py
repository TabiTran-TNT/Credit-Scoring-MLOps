import json
import joblib
import numpy as np
import pandas as pd
import pytest
from credit_scoring.features import prepare_features
from monitoring.drift_report import psi, build_report, log_window, matured_performance, render_evidently


def test_psi_detects_shift_and_missingness():
    stable = np.linspace(0, 1, 1000)
    assert psi(stable, stable) == pytest.approx(0)
    assert psi(stable, stable + 10) > .2
    assert psi([1] * 100, [1] * 100) == pytest.approx(0)
    assert psi([1] * 100, [2] * 100) > .2
    assert psi([np.nan] * 100, [1] * 100) > .2


def test_log_filter_and_alerts(client, valid_applicant, runtime, tmp_path):
    from app.model_loader import get_model_id
    client.post('/predict', json=valid_applicant)
    rows, errors, malformed = log_window(runtime['log_file'], get_model_id())
    assert len(rows) == 1
    assert log_window(runtime['log_file'], 'different-model')[0] == []
    with runtime['log_file'].open('a') as stream:
        stream.write('invalid JSON\n')
    assert log_window(runtime['log_file'], get_model_id())[2] == 1
    with pytest.raises(ValueError, match='timezone'):
        log_window(runtime['log_file'], get_model_id(), since='2020-01-01')


def test_report_detects_shift_without_claiming_performance(model_artifacts, synthetic_data, tmp_path):
    model = joblib.load(model_artifacts['model'])
    raw, _ = synthetic_data
    reference = prepare_features(raw)
    current = reference.copy()
    current['AMT_INCOME_TOTAL'] *= 100
    scores = model.predict_proba(current)[:, 1]
    records = [{'request_id': str(i), 'features': current.iloc[i].to_dict(),
                'outputs': {'probability_of_default': float(score)}, 'inference_time_ms': 5}
               for i, score in enumerate(scores)]
    report = build_report(reference, records, model, 'test-model', tmp_path, render=False)
    assert 'feature_drift' in report['alerts']
    assert report['performance']['status'] == 'labels_not_supplied'
    assert (tmp_path / 'alerts.jsonl').exists()


def test_mature_labels_exclude_future_and_immature(tmp_path):
    records = [{'request_id': str(i), 'outputs': {'probability_of_default': .9 if i % 2 else .1}}
               for i in range(24)]
    labels = pd.DataFrame({'request_id': [str(i) for i in range(24)], 'TARGET': [i % 2 for i in range(24)],
                           'label_available_at': ['2020-01-01T00:00:00Z'] * 24, 'is_mature': [1] * 24})
    path = tmp_path / 'labels.csv'
    labels.loc[0, 'label_available_at'] = '2099-01-01T00:00:00Z'
    labels.loc[1, 'is_mature'] = 0
    labels.to_csv(path, index=False)
    # read_csv would infer numeric IDs; force string IDs by using nonnumeric keys.
    labels['request_id'] = 'r-' + labels['request_id']
    for row in records:
        row['request_id'] = 'r-' + row['request_id']
    labels.to_csv(path, index=False)
    report = matured_performance(records, path, as_of='2026-01-01T00:00:00Z')
    assert report['rows'] == 22
    assert report['roc_auc'] == 1


def test_evidently_exports_html_and_json(tmp_path):
    reference = pd.DataFrame({'x': np.linspace(0, 1, 100), 'CODE_GENDER': [0, 1] * 50})
    render_evidently(reference, reference.copy(), tmp_path)
    assert (tmp_path / 'evidently.html').stat().st_size > 1000
    assert json.loads((tmp_path / 'evidently.json').read_text())


def test_metrics_endpoint(client, valid_applicant):
    client.post('/predict', json=valid_applicant)
    response = client.get('/metrics')
    assert response.status_code == 200
    assert 'credit_scoring_requests_total' in response.text
    assert 'credit_scoring_handler_seconds_bucket' in response.text
