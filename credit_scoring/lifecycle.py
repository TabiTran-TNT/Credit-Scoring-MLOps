"""Local deployment pointers with integrity checks, locking and audited rollback."""
from datetime import datetime, timezone
import json
from pathlib import Path
from uuid import uuid4
from filelock import FileLock
import joblib
import mlflow.sklearn
from credit_scoring.artifacts import ROOT, atomic_json, canonical_hash, download, registry_client, run_artifact, tree_hash
from credit_scoring.evaluation import champion_version
from credit_scoring.tracking import hash_file


def verify_release(manifest, directory):
    directory = Path(directory).resolve()
    paths = {}
    for key in ('model', 'columns'):
        path = (directory / manifest[key + '_path']).resolve()
        if not path.is_relative_to(directory) or not path.is_file():
            raise ValueError('Release path escapes deployment directory or is missing')
        if hash_file(path) != manifest[key + '_sha256']:
            raise ValueError('Release fingerprint mismatch')
        paths[key] = path
    return paths


def _activate(client, name, target, directory, action):
    pointer = directory / 'current.json'
    previous = json.loads(pointer.read_text()) if pointer.exists() else None
    old_alias = champion_version(client, name)
    try:
        client.set_registered_model_alias(name, 'champion', target['version'])
        atomic_json(pointer, target)
        with (directory / 'history.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps({'action': action, 'at': datetime.now(timezone.utc).isoformat(),
                                     'from': previous, 'to': target}) + '\n')
    except Exception:
        if old_alias is None:
            client.delete_registered_model_alias(name, 'champion')
        else:
            client.set_registered_model_alias(name, 'champion', old_alias)
        if previous is not None:
            atomic_json(pointer, previous)
        else:
            pointer.unlink(missing_ok=True)
        raise
    return target


def promote_model(name, version, directory=None):
    directory = Path(directory or ROOT / 'deployments').resolve()
    directory.mkdir(parents=True, exist_ok=True)
    client = registry_client()
    version = str(version)
    with FileLock(str(directory / '.promotion.lock'), timeout=30):
        candidate = client.get_model_version(name, version)
        if candidate.tags.get('validation_status') != 'passed':
            raise ValueError('Model has not passed validation')
        report = json.loads(run_artifact(client, candidate.tags['validation_run_id'], 'validation/report.json').read_text())
        if canonical_hash(report) != candidate.tags.get('validation_report_sha256'):
            raise ValueError('Validation report fingerprint mismatch')
        if not report['passed'] or report['model_name'] != name or report['version'] != version:
            raise ValueError('Validation report does not authorize this model')
        age = (datetime.now(timezone.utc) - datetime.fromisoformat(report['created_at'])).total_seconds() / 3600
        if age < 0 or age > report['rules']['max_report_age_hours']:
            raise ValueError('Validation report expired; validate again')
        if champion_version(client, name) != report['baseline_version']:
            raise ValueError('Champion changed since validation; validate again')
        artifact = download(f'models:/{name}/{version}')
        if tree_hash(artifact) != report['model_sha256']:
            raise ValueError('Model changed after validation')
        model = mlflow.sklearn.load_model(str(artifact))
        release = directory / 'releases' / uuid4().hex
        release.mkdir(parents=True)
        model_path, columns_path = release / 'model.pkl', release / 'columns.pkl'
        joblib.dump(model, model_path)
        joblib.dump(list(model.feature_names_in_), columns_path)
        target = {
            'schema_version': 1, 'model_name': name, 'version': version,
            'model_uri': f'models:/{name}/{version}',
            'validation_id': report['validation_id'],
            'evaluation_contract': report['evaluation_contract'],
            'model_path': model_path.relative_to(directory).as_posix(),
            'columns_path': columns_path.relative_to(directory).as_posix(),
            'model_sha256': hash_file(model_path), 'columns_sha256': hash_file(columns_path),
        }
        atomic_json(release / 'release.json', target)
        return _activate(client, name, target, directory, 'promote')


def rollback_model(name, version, directory=None):
    directory = Path(directory or ROOT / 'deployments').resolve()
    client = registry_client()
    with FileLock(str(directory / '.promotion.lock'), timeout=30):
        events = [json.loads(line) for line in (directory / 'history.jsonl').read_text().splitlines()]
        eligible = [e['to'] for e in events if e['to']['model_name'] == name and e['to']['version'] == str(version)]
        if not eligible:
            raise ValueError('Rollback target must have been deployed previously')
        target = eligible[-1]
        verify_release(target, directory)
        return _activate(client, name, target, directory, 'rollback')
