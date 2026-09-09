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
