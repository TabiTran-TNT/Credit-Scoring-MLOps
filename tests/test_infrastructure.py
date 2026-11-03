from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]


def test_compose_mounts_models_and_has_ready_dependency():
    config = yaml.safe_load((ROOT / 'compose.yaml').read_text())
    assert config['services']['tools']['depends_on']['mlflow']['condition'] == 'service_healthy'
    assert './model:/workspace/model:ro' in config['services']['api']['volumes']
    assert config['services']['mlflow']['command'].count('--workers') == 1
    assert '127.0.0.1:8000:8000' in config['services']['api']['ports']


def test_ci_uses_single_requirements_and_never_retrains_for_publish():
    workflow = yaml.load((ROOT / '.github/workflows/ci-cd.yml').read_text(), Loader=yaml.BaseLoader)
    test_steps = workflow['jobs']['test']['steps']
    assert any('pip install -r requirements.txt' in s.get('run', '') for s in test_steps)
    steps = workflow['jobs']['build']['steps']
    publishes = [step for step in steps if 'docker push' in step.get('run', '')]
    assert len(publishes) == 1
    assert 'workflow_dispatch' in publishes[0]['if']
    assert 'inputs.publish' in publishes[0]['if']
    assert not any('train_model' in step.get('run', '') for step in steps)
    assert 'COPY model' not in (ROOT / 'Dockerfile').read_text()
    assert '.env' in (ROOT / '.dockerignore').read_text().splitlines()


def test_one_requirements_file_and_demo_is_explicitly_synthetic(tmp_path):
    from scripts.create_demo_artifacts import create_demo_artifacts
    import json
    create_demo_artifacts(tmp_path)
    assert json.loads((tmp_path / 'metadata.json').read_text())['synthetic'] is True
    assert not (ROOT / 'requirements-dev.txt').exists()
    assert not (ROOT / 'requirements-mlflow.txt').exists()
