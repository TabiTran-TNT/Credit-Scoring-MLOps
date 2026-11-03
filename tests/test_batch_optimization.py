import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from optimization.benchmark import benchmark
from scripts.batch_score import batch_score


def test_batch_api_parity_and_atomic_failure(client, runtime, valid_applicant, tmp_path):
    raw = pd.DataFrame([{**valid_applicant, 'SK_ID_CURR': i} for i in range(5)])
    source, destination = tmp_path / 'raw.csv', tmp_path / 'scores.csv'
    raw.to_csv(source, index=False)
    assert batch_score(source, destination, chunk_size=2) == 5
    results = pd.read_csv(destination)
    response = client.post('/predict', json=valid_applicant).json()
    assert list(results['row_id']) == list(range(5))
    np.testing.assert_allclose(results['probability_of_default'], response['probability_of_default'])
    before = destination.read_bytes()
    raw.loc[4, 'AMT_INCOME_TOTAL'] = -1
    raw.to_csv(source, index=False)
    with pytest.raises(ValueError):
        batch_score(source, destination, chunk_size=2)
    assert destination.read_bytes() == before


def test_onnx_export_has_full_dataset_parity(model_artifacts, synthetic_data, tmp_path):
    from credit_scoring.features import prepare_features
    raw, _ = synthetic_data
    features_file = tmp_path / 'features.csv'
    prepare_features(raw).to_csv(features_file, index=False)
    report = benchmark(model_artifacts['model'], features_file, tmp_path / 'onnx', repeats=5)
    assert report['parity_rows'] == len(raw)
    assert report['decision_parity']
    assert (tmp_path / 'onnx/model.onnx').is_file()
    assert (tmp_path / 'onnx/profile.txt').is_file()
