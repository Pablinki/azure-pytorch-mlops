# Decisions (ADR-lite)

One entry per decision that a reviewer might question. Format: Context / Decision / Consequence.
The dependency table at the end justifies every third-party package in one line.

## D1. TextCNN in plain PyTorch instead of a transformer fine-tune
- **Context:** the project must show PyTorch engineering, train on CPU in minutes and stay cheap on Azure.
- **Decision:** a Kim-2014 TextCNN (`nn.Embedding` → three parallel `Conv1d` → max-over-time pooling → linear).
- **Consequence:** ~90 % accuracy on AG News instead of ~94 % for DistilBERT, but a real training loop
  (AMP, grad clipping, checkpoints, early stopping) that we own end to end and that fits a `Standard_DS3_v2`.

## D2. One package, one CLI, one config schema
- **Context:** MLOps repos rot when the training script, the notebook and the API each parse their own config.
- **Decision:** `src/textclf` is the only code; `textclf` (typer) is the only entrypoint; `Config` (pydantic,
  `extra="forbid"`) is the only hyperparameter schema and it is serialised into the model artifacts.
- **Consequence:** the API rebuilds the exact model from `config.json`; a typo in YAML fails before compute starts.

## D3. Exception hierarchy with three catch boundaries
- **Context:** blind `except Exception` and log-and-re-raise chains make production failures unreadable.
- **Decision:** every package error derives from `MLProjectError` (see `src/textclf/exceptions.py`); third-party
  errors are wrapped with `from e` at module boundaries; only the CLI, the API and the MLflow run context catch.
  ruff rule sets `BLE` and `TRY` enforce it.
- **Consequence:** one log line per failure with structured `context`, stable CLI exit codes, HTTP status mapping
  in one place.

## D4. `tenacity` at the operation level only
- **Context:** Azure SDKs already retry individual HTTP calls; nesting retries multiplies latency.
- **Decision:** `transient_retry` wraps whole operations (download + verify, dataset fetch) and only retries
  connection/timeout errors and HTTP 408/429/5xx.
- **Consequence:** validation, auth and programming errors surface on the first attempt.

## D5. `uv` for dependency resolution and locking
- **Context:** reproducibility needs a lock file; CI and the AML image install from `pyproject.toml`.
- **Decision:** `uv.lock` pins the developer environment; images use `pip install` from `pyproject.toml` floors
  plus a CPU-only torch index so the image stays small.
- **Consequence:** two resolution paths (lock locally, floors in images). Accepted: the images are rebuilt from
  scratch per SHA and their torch build must differ (CPU wheel) anyway.

## D6. MLflow local backend is `sqlite:///mlruns.db`, not `./mlruns`
- **Context:** MLflow 3 raises on the legacy filesystem tracking store unless `MLFLOW_ALLOW_FILE_STORE=true`.
- **Decision:** `configure_tracking()` uses `MLFLOW_TRACKING_URI` when set (Azure ML exports it inside jobs),
  respects an explicit `mlflow.set_tracking_uri`, and otherwise defaults to a sqlite file in the working dir.
- **Consequence:** one code path locally and in the cloud; `mlruns.db` and `mlartifacts/` are git-ignored.

## D7. Artifacts are attached to the run by id, after the run context closes
- **Context:** `run_training` reloads `best.pt` and exports artifacts after `Trainer.fit()` returns, i.e. after
  the `with mlflow.start_run()` block. The fluent `mlflow.log_artifacts` would then open an implicit run and
  leave it active, breaking the next training in the same process.
- **Decision:** `Trainer.fit` records `run_id`; `export_artifacts` logs through `MlflowClient().log_artifacts(run_id, ...)`.
- **Consequence:** no hidden runs; a test asserts that no run leaks between trainings.

## D8. Signal handlers are installed inside `fit()` and restored afterwards
- **Context:** the plan registers SIGTERM/SIGINT handlers in `Trainer.__init__`; that leaks the handler into the
  host process (tests, notebooks) and fails outside the main thread.
- **Decision:** install in `fit()`, restore the previous handlers in a `finally`, ignore `ValueError` off-thread.
- **Consequence:** identical preemption behaviour on Azure ML, no side effects on the caller.

## D9. MLflow through `azureml-mlflow`, not the AML SDK logging API
- **Context:** the training loop must log identically on a laptop and inside an Azure ML job.
- **Decision:** plain `mlflow.log_*` calls; the `azureml-mlflow` plugin resolves the `azureml://` tracking URI
  that the job exports. Locally the same calls hit sqlite.
- **Consequence:** no Azure imports in `train.py`; the run shows up in Studio with curves and artifacts.

## D10. Bake the model into the serving image instead of downloading at startup
- **Context:** the API scales to zero, so cold-start time and runtime credentials matter.
- **Decision:** CI downloads the registered model version and copies it into `/app/model`; the image tag is
  the git SHA, the model version is a workflow input/variable.
- **Consequence:** immutable, reproducible artifact; no storage credentials in the container; a new model
  means a new image (that is the point).

## D11. Azure Container Apps instead of an AML managed online endpoint
- **Context:** both can serve the model; the endpoint is simpler to wire but bills a VM 24/7.
- **Decision:** Container Apps with a KEDA HTTP-concurrency rule and `minReplicas: 0`.
- **Consequence:** ~0 cost when idle, plain FastAPI/Docker skills, cold starts of a few seconds (D16).

## D12. Bicep instead of Terraform
- **Context:** one cloud, one team, a portfolio project.
- **Decision:** Bicep, deployed with `az deployment group create`; `what-if` as the plan step.
- **Consequence:** no state backend to secure; ARM-native typing; not portable to other clouds (acceptable).

## D13. `az acr build` instead of Docker on the runner
- **Context:** the image is ~1 GB with CPU torch; building on the GitHub runner then pushing is slow.
- **Decision:** the runner sends the build context to ACR Tasks; the image never leaves Azure.
- **Consequence:** no Docker daemon needed in CI; local `docker build` stays available for debugging.

## D14. OIDC federated credential instead of a service-principal secret
- **Context:** a client secret in GitHub is a long-lived cloud credential waiting to leak or expire.
- **Decision:** `azure/login` exchanges the workflow's OIDC token for an Azure token; the federated credential
  is bound to `repo:Pablinki/azure-pytorch-mlops:ref:refs/heads/main`.
- **Consequence:** GitHub stores only ids (client, tenant, subscription), which are not secrets.

## D15. Two Dockerfiles instead of one multi-target file
- **Context:** the training image needs the AML base + MLflow + datasets; the serving image needs FastAPI only.
- **Decision:** `aml/Dockerfile.train` (deps only, code installed by the job) and `Dockerfile` (serve).
- **Consequence:** two small, readable files; the serving image does not carry pandas/mlflow/sklearn.

## D16. Scale-to-zero: cold starts are the price of ~0 idle cost
- **Context:** `minReplicas: 0` means the first request after idling waits for a container start plus model load.
- **Decision:** accept it for a portfolio/dev deployment; keep the image small, warm up the model in
  `Predictor.from_dir`, and document `minReplicas: 1` as the production knob.
- **Consequence:** measured in `docs/evidence/` (cold-start latency vs warm p95).

## D17. The AML cluster pulls images with a managed identity, not the ACR admin user
- **Context:** the first smoke job failed with "Failed to pull Docker image ... ensure the ACR has Admin user
  enabled or a Managed Identity with AcrPull". `adminUserEnabled: false` is deliberate (no registry passwords).
- **Decision:** `aml/compute.yaml` gives `cpu-cluster` a system-assigned identity and `make compute` grants it
  `AcrPull` on the workspace ACR; the container app already does the same for serving.
- **Consequence:** every image pull in the project (training and serving) is identity-based; the admin user stays off.

## Dependency justification

| Package | Why |
|---|---|
| `torch` | the model, the training loop and inference |
| `numpy` | seeding and array plumbing between torch and sklearn |
| `pydantic` | typed, `extra="forbid"` config and request/response schemas |
| `pydantic-settings` | serving settings from `TEXTCLF_*` env vars with validation |
| `pyyaml` | read `configs/train.yaml` |
| `typer` | one CLI with typed options and help text; the CLI error boundary |
| `tenacity` | bounded exponential-backoff retry with jitter (D4); a hand-rolled version is 30+ lines with worse logging |
| `datasets` (train) | download AG News from the Hugging Face hub with caching |
| `pandas` + `pyarrow` (train) | write/read parquet splits with typed columns |
| `scikit-learn` (train) | accuracy, macro-F1, per-class report, stratified split |
| `mlflow` + `azureml-mlflow` (train) | metric/artifact tracking that works identically locally and inside AML jobs |
| `fastapi` + `uvicorn` (serve) | typed HTTP API with automatic validation (422) and OpenAPI docs |
| `azure-ai-ml` + `azure-identity` (azure) | SDK v2 pipeline script (stretch), `DefaultAzureCredential` |
| `pytest`, `pytest-cov`, `httpx` (dev) | tests, coverage gate, FastAPI `TestClient` transport |
| `ruff`, `mypy`, `types-PyYAML`, `pre-commit` (dev) | lint (incl. BLE/TRY), strict typing, git hooks |
