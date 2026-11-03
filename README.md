# Credit Scoring MLOps

An ML pipeline for the Home Credit Default Risk dataset, with experiment tracking,
model validation, versioned releases, online and batch scoring, and monitoring.

The classifier estimates the probability of payment difficulty using 18 features
derived from application data. A scikit-learn pipeline contains median imputation
and gradient boosting. Imputation is fitted on training rows only; validation and
inference reuse those fitted values.

## Features

| Component | Purpose |
|---|---|
| Shared preprocessing | Applies the same transformations to training, API and batch data |
| Training | Uses a stratified 70/15/15 split and keeps outputs separate for each run |
| MLflow | Records parameters, metrics, dataset/code hashes and registered model versions |
| Validation | Checks schema, probabilities, ROC-AUC/AP, latency and champion regression |
| Promotion | Verifies the validation report and artifacts before selecting a version |
| Releases and rollback | Maintains a pinned local release, atomic pointer and deployment history |
| FastAPI | Loads the selected model at startup and validates application input |
| Batch scoring | Scores CSV chunks and replaces output only after every chunk succeeds |
| Monitoring | Produces feature/score PSI, missingness checks, mature-label metrics and Evidently reports |
| Optimization | Profiles inference and checks ONNX probability/decision parity before export |
| Delivery | Provides a non-root Docker image, Compose services and GitHub Actions workflow |

## Installation

Use Python 3.12. All Python dependencies, including test and monitoring packages,
are in `requirements.txt`. Run commands from the repository root.

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pip check
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
```

Linux/macOS:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip check
test -e .env || cp .env.example .env
```

The pandas 2.3.3 pin is intentional: Evidently 0.7.23 imports an API removed in
pandas 3. Keep existing `.env` settings when updating the project. Explicit
environment variables take precedence over `.env`.

## Data and configuration

Download `application_train.csv` from the
[Home Credit competition](https://www.kaggle.com/competitions/home-credit-default-risk/data)
under its access terms and place it at `data/application_train.csv`. Data, models,
logs, registry state and `.env` are ignored by Git.

| Variable | Default | Purpose |
|---|---|---|
| `MLFLOW_ENABLED` | `true` | Enable training run tracking |
| `MLFLOW_TRACKING_URI` | `http://127.0.0.1:5000` | Tracking and registry connection |
| `MLFLOW_EXPERIMENT_NAME` | `credit-scoring` | Experiment grouping |
| `MLFLOW_REGISTERED_MODEL_NAME` | `credit-scoring` | Registered model name |
| `DEPLOYMENT_MANIFEST` | Unset | Optional pinned release selected for serving |

Without a deployment manifest, serving uses `model/credit_model.pkl` and
`model/feature_columns.pkl`. `MODEL_PATH`, `FEATURES_PATH` and `LOG_DIR` can
override the legacy artifact paths and log directory.

## Training

Start MLflow in one terminal:

```powershell
python -m scripts.start_mlflow
```

Open `http://127.0.0.1:5000`. Metadata is stored in `.mlflow/mlflow.db`, and
artifacts in `.mlflow/artifacts/`. In another terminal, activate the environment:

```powershell
python -m scripts.train_model --n-estimators 100 --max-depth 3 --run-name baseline
```

Each run records the dataset fingerprint, split seed, feature code fingerprint,
Git revision/working-tree state, parameters and validation metrics. Its registered
artifact includes the fitted pipeline, signature and dependency versions. The
final test set is saved without being used for model selection.

```text
model/runs/<run_id>/
    credit_model.pkl
    feature_columns.pkl
    training_metadata.json
data/runs/<run_id>/
    reference_data.csv
    validation_data.csv
    validation_labels.csv
    test_data.csv
    test_labels.csv
```

Training does not overwrite the legacy serving files. Run training again with
different parameters to compare configurations in MLflow. Version numbers reflect
registration order, not model quality. Set `MLFLOW_ENABLED=false` for offline
training; outputs still receive a local run ID, without a registry version.

## Validation and promotion

The following examples use version `3`. Use the version printed by your training
command instead.

```powershell
python -m scripts.validate_model --version 3
python -m scripts.promote_model --version 3
```

Rules in `configs/validation.yaml` cover minimum sample count, ROC-AUC/AP,
single-row p95 latency and permitted AUC regression. Candidate/champion comparisons
require matching dataset, split, feature code and evaluation-file fingerprints.
Models registered before this evaluation metadata was introduced need retraining
before using the gate.

Reports are saved under `reports/validation/` and in MLflow. Promotion requires a
passing report that has not expired, an unchanged artifact and the same champion
baseline used during validation. It creates a local release, writes
`deployments/current.json` and updates the registry's `champion` alias.

Gate values are example defaults and need calibration for a real application.
Use validation data for selection and keep the final test set separate.

## API

To serve a promoted release in PowerShell:

```powershell
$env:DEPLOYMENT_MANIFEST = 'deployments/current.json'
python -m uvicorn app.main:app --reload
```

On Linux/macOS, use `export DEPLOYMENT_MANIFEST=deployments/current.json` before
starting Uvicorn. The setting can also be added to `.env`.

| Endpoint | Purpose |
|---|---|
| `GET /health` | Check whether the model is loaded |
| `POST /predict` | Score one application |
| `GET /metrics` | Prometheus request counts and handler-latency histograms |
| `GET /docs` | Interactive API documentation |

Open `http://127.0.0.1:8000/docs` and use
[`examples/request.json`](examples/request.json), or send a PowerShell request:

```powershell
Invoke-RestMethod -Uri http://127.0.0.1:8000/predict `
  -Method Post -ContentType 'application/json' `
  -Body (Get-Content examples/request.json -Raw)
```

Input uses raw uppercase CSV names, such as `DAYS_BIRTH`, `CODE_GENDER` and
`AMT_CREDIT`. Do not send `TARGET`. Every field is required; nullable fields must
be present as JSON `null`. The fitted imputer handles missing numeric values.
Invalid input returns 422.

The response contains `prediction`, `probability_of_default` and `risk_category`.
The binary threshold is 0.5; category boundaries are 0.3 and 0.6. These are example
rules, not a calibrated lending policy. Headers identify the request, model
fingerprint and registered version when a manifest is selected.

A model loads once per process. Restart/redeploy the API after promotion or
rollback; changing an alias does not replace a model already in memory.

## Batch scoring

Use the same model selection as the API:

```powershell
python -m scripts.batch_score --input data/new_applications.csv --output reports/scores.csv --chunk-size 1000
```

Input needs the same raw feature fields. Optional `SK_ID_CURR` becomes the output
row ID; otherwise the input row index is used. Extra source columns do not enter
the model. Output contains the ID, prediction, probability, category and model
identity. If a chunk fails, an existing output is preserved.

## Logs and monitoring

`logs/predictions.jsonl` records results, request/model identity, handler time and
features before imputation. Missing values become JSON `null`. Lifecycle and
error events use the same format. Logs rotate at about 10 MiB with five backups;
use a collector instead of a shared rotating file when scaling to multiple workers.

Use reference data from the run that produced the serving model. Replace `RUN_ID`:

```powershell
python -m scripts.monitor_model --reference data/runs/RUN_ID/reference_data.csv --logs logs/predictions.jsonl --output reports/monitoring
```

Use `--since 2026-11-01T00:00:00Z` for a timezone-aware start time and `--max-rows`
to bound the event window. The command reads one specified log file; pass a rotated
file explicitly when needed. Records are filtered by the loaded model fingerprint.

Outputs include `summary.json`, `alerts.jsonl`, `evidently.html` and
`evidently.json`. They cover feature/score PSI, missingness, prediction errors and
latency. Exit code 2 means investigation is required. Drift alone does not prove
performance degradation and does not automatically retrain or roll back models.
Scheduling and external alert delivery are not configured by default.

For labeled evaluation, pass `--labels data/mature_labels.csv`. Required columns
are `request_id`, `TARGET`, `label_available_at` (UTC) and `is_mature` (boolean or
0/1). IDs must be unique. Only mature labels available by the evaluation time
participate; ROC-AUC/AP require at least 20 matched rows and both classes.

## Profiling and ONNX

```powershell
python -m optimization.benchmark --model model/runs/RUN_ID/credit_model.pkl --validation data/runs/RUN_ID/validation_data.csv --output reports/optimization
```

Input must contain engineered features in the model's expected order. Export
checks probabilities and binary decisions on every supplied row before saving
`model.onnx`. The output directory also receives `profile.txt` and `benchmark.json`,
including parity tolerances and warmed single-row mean/p95 timings.

Timings exclude HTTP, logging and raw feature preparation. The API continues to
use the native pipeline; an ONNX export is not activated automatically.

## Recovery

```powershell
python -m scripts.rollback_model --version 2
```

The target must be a previously deployed release with intact files. Restart the
API and verify `X-Model-Version`/`X-Model-ID` afterward.

- **422:** inspect request fields, types and ranges.
- **500:** find the request ID in the JSON log and inspect its traceback.
- **Fingerprint mismatch:** restore the approved release or roll back.
- **Expired/stale validation:** validate again against the current champion.
- **Drift alert:** inspect upstream changes and missingness; evaluate mature labels
  before deciding to retrain.

For an environment migration, transfer the pinned release with the same code and
dependency versions. Verify fingerprints, start the destination on a separate
port and compare representative scores before switching traffic. Keep the old
release for recovery. Local locking is not distributed coordination, and default
restarts have a brief availability gap. Stop SQLite writers before backing up
metadata and artifacts; do not copy an active database.

## Tests

```powershell
python -m pytest tests -v --basetemp work/pytest
python -m pip check
```

Tests use small synthetic datasets and temporary model/registry/log directories.
They do not retrain the complete dataset or overwrite serving artifacts. No
Uvicorn or MLflow server is needed. The optional local-artifact test skips when
model files are absent.

## Docker and Compose

Docker Desktop must be installed with its Linux engine running. The Python Docker
SDK does not provide the engine and is not required for these CLI commands.

```powershell
docker version
docker compose version
docker compose config --quiet
docker compose build
docker compose up -d mlflow
docker compose run --rm tools python -m scripts.train_model --run-name container-baseline
docker compose run --rm tools python -m scripts.validate_model --version 3
docker compose run --rm tools python -m scripts.promote_model --version 3
```

Use the version registered in this MLflow store, which is separate from host-native
MLflow. Create writable `data`, `model`, `deployments`, `logs`, `reports` and `work`
directories first. Set `DEPLOYMENT_MANIFEST=deployments/current.json` in `.env`:

```powershell
docker compose up -d api
docker compose logs -f api
```

Stop local services first if ports 8000/5000 are occupied. API/MLflow use UID 1000;
mounts must permit their required access. On Linux, set `LOCAL_UID` and `LOCAL_GID`
for the tools service to match the host user. MLflow state uses a named volume.
Dataset, secrets and production models are excluded from the image; models and
releases are supplied through mounts.

## CI/CD

GitHub Actions runs tests, builds the image and smoke-tests it with a temporary
synthetic model. It does not require the competition dataset. Publishing to GHCR
requires a manual workflow request on `main` with `publish` enabled. It publishes
the image tested by that build job, without the synthetic model, and does not
deploy a production release. A remote repository and workflow execution are
required to use this pipeline.

## Evaluation and limitations

A local run with 100 trees and depth 3 used 215,257 training rows and 46,127
validation rows. Validation ROC-AUC was **0.754984**, with average precision
**0.242130**. Another 46,127 rows were reserved for final testing. ONNX probability
and decision parity was checked on every validation row. These are validation
results on a static dataset, not production performance guarantees.

The project starts from the
[Home Credit MLOps tutorial](https://hackernoon.com/weekend-project-i-built-a-full-mlops-pipeline-for-a-credit-scoring-model-and-you-can-too).
The static split does not establish event-time correctness, an online feature
store or near-real-time ingestion. Demographic attributes and proxies need
separate review before real lending use. Local filesystem/registry access is
trusted; authentication, authorization, retention and centralized logging belong
to the deployment environment. Feature logs contain financial/demographic data
and should not be published with the source code.
