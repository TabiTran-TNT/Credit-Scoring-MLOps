"""Content identities and atomic local metadata writes."""
from hashlib import sha256
import json
import os
from pathlib import Path
from uuid import uuid4
from credit_scoring.tracking import hash_file

ROOT = Path(__file__).resolve().parents[1]


def canonical_hash(value):
    return sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def tree_hash(directory):
    directory = Path(directory)
    return canonical_hash({p.relative_to(directory).as_posix(): hash_file(p)
                           for p in sorted(directory.rglob('*')) if p.is_file()})


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.' + uuid4().hex + '.tmp')
    try:
        temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def registry_client():
    import mlflow
    from credit_scoring.config import get_tracking_config
    config = get_tracking_config()
    mlflow.set_tracking_uri(config.tracking_uri)
    mlflow.set_registry_uri(config.tracking_uri)
    return mlflow.MlflowClient()


def download(uri):
    import mlflow.artifacts
    destination = ROOT / 'work' / 'downloads' / uuid4().hex
    destination.mkdir(parents=True)
    return Path(mlflow.artifacts.download_artifacts(artifact_uri=uri, dst_path=str(destination)))


def run_artifact(client, run_id, path):
    destination = ROOT / 'work' / 'downloads' / uuid4().hex
    destination.mkdir(parents=True)
    return Path(client.download_artifacts(run_id, path, dst_path=str(destination)))
