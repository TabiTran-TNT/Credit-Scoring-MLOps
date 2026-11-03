"""Start a local MLflow tracking/registry server on port 5000."""

from pathlib import Path
import subprocess
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def server_command():
    state_dir = PROJECT_ROOT / ".mlflow"
    state_dir.mkdir(parents=True, exist_ok=True)
    artifacts = state_dir / "artifacts"
    artifacts.mkdir(exist_ok=True)
    database_uri = "sqlite:///" + (state_dir / "mlflow.db").as_posix()
    return [
        sys.executable, "-m", "mlflow", "server",
        "--host", "127.0.0.1", "--port", "5000",
        "--workers", "1",
        "--backend-store-uri", database_uri,
        "--serve-artifacts", "--artifacts-destination", artifacts.as_uri(),
    ]


if __name__ == "__main__":
    print("Starting MLflow at http://127.0.0.1:5000", flush=True)
    raise SystemExit(subprocess.call(server_command(), cwd=PROJECT_ROOT))
