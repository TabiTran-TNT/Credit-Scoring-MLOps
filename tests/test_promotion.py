import json
from pathlib import Path
import pytest
from test_tracking import tracking_environment
from credit_scoring.evaluation import ValidationRules, validate_model
from credit_scoring.lifecycle import promote_model, rollback_model, verify_release


def rules(**kwargs):
    return ValidationRules(min_rows=10, min_roc_auc=0, min_average_precision=0,
                           max_p95_latency_ms=10000, latency_repeats=5, **kwargs)


def test_validate_promote_reload_rollback_and_stale_guard(tracking_environment, monkeypatch, tmp_path):
    from app import model_loader
    train, client, _, _ = tracking_environment
    first = train.main(n_estimators=8, max_depth=2)
    second = train.main(n_estimators=12, max_depth=3)
    name = 'test-credit-scoring'
    deployment = tmp_path / 'deployment'
    with pytest.raises(ValueError, match='not passed'):
        promote_model(name, '1', deployment)
    assert validate_model(name, '1', rules())['passed']
    # This report becomes stale after the first champion is selected.
    assert validate_model(name, '2', rules())['passed']
    promoted = promote_model(name, '1', deployment)
    monkeypatch.setenv('DEPLOYMENT_MANIFEST', str(deployment / 'current.json'))
    model_loader.load_model()
    assert model_loader.get_model_version() == '1'
    with pytest.raises(ValueError, match='Champion changed'):
        promote_model(name, '2', deployment)
    assert validate_model(name, '2', rules())['baseline_version'] == '1'
    promote_model(name, '2', deployment)
    assert str(client.get_model_version_by_alias(name, 'champion').version) == '2'
    rollback_model(name, '1', deployment)
    assert json.loads((deployment / 'current.json').read_text())['version'] == '1'
    assert str(client.get_model_version_by_alias(name, 'champion').version) == '1'
    with pytest.raises(ValueError, match='previously'):
        rollback_model(name, '999', deployment)
    paths = verify_release(promoted, deployment)
    paths['model'].write_bytes(b'tampered')
    with pytest.raises(ValueError, match='fingerprint'):
        model_loader.load_model()


def test_failed_quality_gate_blocks_promotion(tracking_environment, tmp_path):
    train, client, _, _ = tracking_environment
    train.main(n_estimators=8, max_depth=2)
    strict = ValidationRules(min_rows=1000000, latency_repeats=5)
    result = validate_model('test-credit-scoring', '1', strict)
    assert not result['passed']
    assert client.get_model_version('test-credit-scoring', '1').tags['validation_status'] == 'failed'
    with pytest.raises(ValueError, match='not passed'):
        promote_model('test-credit-scoring', '1', tmp_path / 'deployment')


def test_release_paths_cannot_escape_directory(tmp_path):
    with pytest.raises(ValueError, match='escapes'):
        verify_release({'model_path': '../outside.pkl'}, tmp_path)
