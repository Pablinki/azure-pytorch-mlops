# textclf tutorial: set up, develop, operate

This is the hands-on companion to the [README](../README.md). The README says *what* the project is; this
document walks you through *doing* things with it, in the order you will actually need them. Every command
here was run at least once while building the project (see [evidence/](evidence/)).

Contents

1. [The mental model](#1-the-mental-model)
2. [One-time setup on your machine](#2-one-time-setup-on-your-machine)
3. [Developing locally](#3-developing-locally)
4. [Provisioning Azure (once)](#4-provisioning-azure-once)
5. [Training in the cloud](#5-training-in-the-cloud)
6. [Deploying and operating the API](#6-deploying-and-operating-the-api)
7. [How CI/CD works and how to repair it](#7-how-cicd-works-and-how-to-repair-it)
8. [Day-2 operations: cost, teardown, upgrades](#8-day-2-operations-cost-teardown-upgrades)
9. [Troubleshooting: every error we hit](#9-troubleshooting-every-error-we-hit)
10. [Command cheat sheet](#10-command-cheat-sheet)

---

## 1. The mental model

There are three loops. You will spend most of your time in the first one.

| Loop | Where it runs | Entry point | Costs money? |
|---|---|---|---|
| **Develop**: change code, run tests, train a small model, hit the API on `localhost` | your laptop | `make check`, `make train-local`, `make serve` | no |
| **Train**: run the same code on a bigger machine, track metrics, register the model | Azure ML (`mlw-textclf`) | `make submit`, `make register` | cents per run |
| **Serve**: bake a registered model into an image and run it as an autoscaling HTTP service | Azure Container Apps (`textclf-api`) | push to `main` (CI) or `make deploy` | ~0 when idle |

The same Python package (`src/textclf`) runs in all three. That is the point: nothing is rewritten for the
cloud, the cloud just supplies data paths, a bigger CPU and an MLflow endpoint.

```
data/ag_news ──> textclf train ──> outputs/model/{model.pt,vocab.json,labels.json,config.json,metrics.json}
                                          │
                       local: textclf serve reads it        cloud: registered as textclf:<version>
                                          │                              │
                                   http://127.0.0.1:8000      CI bakes it into textclf-api:<git sha>
                                                                          │
                                                             https://textclf-api.<env>.azurecontainerapps.io
```

Five files make a model. If those five files exist in a folder, `Predictor.from_dir(folder)` can serve it.
Everything else (checkpoints, MLflow runs, images) is derived.

---

## 2. One-time setup on your machine

### 2.1 Tools

| Tool | Why | Install |
|---|---|---|
| `uv` | Python + dependency manager; creates `.venv` from `uv.lock` | `pip install -U uv` (or the installer from astral.sh) |
| Python 3.11 | the version the lock file and the Docker images use | `uv python install 3.11` (uv downloads it, no admin rights) |
| `make` | the human interface: `make check`, `make deploy`, ... | Linux/macOS: preinstalled. Windows: `winget install ezwinports.make` |
| Azure CLI + `ml` extension | everything cloud | `winget install Microsoft.AzureCLI` (or brew/apt), then `az extension add -n ml -y` |
| GitHub CLI `gh` | repo, secrets, watching workflow runs | `winget install GitHub.cli`, then `gh auth login` |
| Docker | optional, only for `make docker-build` locally; the cloud builds images in ACR | Docker Desktop |

Verify:

```bash
uv --version && az version --query '"azure-cli"' -o tsv && az extension list --query "[?name=='ml'].version" -o tsv && gh --version && make --version
```

**Windows users, read this once.** Use Git Bash (the Makefile needs `bash`). Two quirks will bite you:

- Git Bash rewrites arguments that start with `/` as Windows paths. Azure resource ids and scopes start with
  `/subscriptions/...`, so prefix those commands with `MSYS_NO_PATHCONV=1` (the Makefile already does).
- The Azure CLI's live log streaming (`az ml job stream`, `az acr build` output) can crash with
  `'charmap' codec can't encode`. The cloud operation keeps running; just check status with `az ... show`
  instead of streaming, or run those commands from PowerShell.

### 2.2 Clone and install

```bash
git clone https://github.com/Pablinki/azure-pytorch-mlops.git
cd azure-pytorch-mlops
make setup          # uv sync --all-extras (creates .venv) + pre-commit install
```

`make setup` takes a few minutes the first time (PyTorch is large). From now on every Python command is run
through `uv run ...` so it uses `.venv` automatically; `make` targets do that for you.

### 2.3 Configuration: `.env`

```bash
cp .env.example .env
```

`.env` is git-ignored and holds only identifiers, never secrets:

| Variable | Meaning | When you fill it |
|---|---|---|
| `AZURE_SUBSCRIPTION_ID` | the subscription to bill | before `az login` / provisioning |
| `AZURE_LOCATION` | region (`eastus2`) | keep |
| `AZURE_RG` | resource group (`rg-textclf`) | keep |
| `AZURE_ML_WORKSPACE` | Azure ML workspace name | keep |
| `AZURE_ACR_NAME` | container registry name | **output of `make provision`**, paste it in |
| `ACA_ENV`, `ACA_APP` | Container Apps environment and app names | keep |
| `TEXTCLF_MODEL_URI` | what `textclf serve` loads locally | keep (`file://outputs/model`) |
| `LOG_FORMAT` | `text` locally, `json` in containers | keep |

The serving process reads `TEXTCLF_*` variables through `ServeSettings` (`src/textclf/config.py`); the Makefile
reads the `AZURE_*` ones with `set -a; source .env; set +a`.

### 2.4 Azure login

```bash
az login                                   # opens a browser
az account set --subscription "$AZURE_SUBSCRIPTION_ID"
az account show --query "{name:name,user:user.name}" -o table
```

If the browser flow hangs or you are on a remote shell, use `az login --use-device-code`: it prints a code you
type at https://login.microsoft.com/device. Make sure you sign in with the account that **owns the
subscription** (a personal account with no subscription logs in fine but sees nothing).

---

## 3. Developing locally

### 3.1 The five-minute loop

```bash
make prepare-data   # downloads AG News once -> data/ag_news/  (120 000 train / 7 600 test rows)
make train-local    # 1 epoch on 5 000 rows, ~15 s on a laptop -> outputs/model/
make serve          # FastAPI on http://127.0.0.1:8000
```

In a second terminal:

```bash
curl -s localhost:8000/ready
curl -s -X POST localhost:8000/predict -H 'content-type: application/json' \
  -d '{"texts":["Stocks rally as tech earnings beat expectations","Real Madrid wins the derby in extra time"]}'
```

You get one prediction per text with a label, a confidence and the full score vector, plus an
`x-request-id` header. Send your own `x-request-id` header and it is echoed back; that id also appears in the
server log line for the request.

A full local training run (5 epochs, all rows, a few minutes on CPU) is `uv run textclf train`.
Evaluate any exported model on the test split:

```bash
uv run textclf evaluate --model-dir outputs/model --data-dir data/ag_news --split test
```

This prints a per-class precision/recall/F1 report and writes `outputs/model/metrics_test.json`.

### 3.2 Where things are

| Path | What lives there | Touch it when |
|---|---|---|
| `configs/train.yaml` | every hyperparameter, mirrors the pydantic defaults | tuning |
| `src/textclf/config.py` | the config schema (`extra="forbid"`) and `ServeSettings` | adding a knob |
| `src/textclf/exceptions.py` | the error hierarchy (`MLProjectError` and children) | adding an error type |
| `src/textclf/retry.py` | `transient_retry` decorator | wrapping a flaky network call |
| `src/textclf/data.py` | download, parquet validation, tokenizer, `Vocab`, batching | data changes |
| `src/textclf/model.py` | `TextCNN` and the `build_model` registry | new architecture |
| `src/textclf/train.py` | `Trainer` (AMP, checkpoints, resume, early stop, MLflow), `run_training` | training behaviour |
| `src/textclf/evaluate.py` | metrics on a loader; the `evaluate` CLI report | new metric |
| `src/textclf/predict.py` | `Predictor`: load the five files, batched inference | inference changes |
| `src/textclf/api/` | FastAPI app (error boundary) and request/response schemas | API changes |
| `src/textclf/cli.py` | the `textclf` command (error boundary) | new command |
| `tests/` | offline tests; `conftest.py` builds a tiny dataset and a tiny trained model | always |
| `aml/`, `infra/`, `.github/workflows/` | Azure ML job files, Bicep, CI/CD | cloud changes |
| `docs/DECISIONS.md` | one entry per non-obvious decision (D1-D18) | any decision |

### 3.3 The quality gate

```bash
make check      # = ruff check + ruff format --check + mypy --strict src + pytest (coverage >= 80 %)
make format     # auto-fix lint and formatting
```

CI runs exactly `make check` plus a Bicep lint. The pre-commit hook (installed by `make setup`) runs ruff and a
few hygiene checks on every commit, and refuses files over 1 MB so checkpoints never get committed.

The test suite is offline and takes about 10 seconds. `tests/conftest.py` creates a 50-row synthetic dataset
whose four classes use disjoint vocabularies, trains the TextCNN for two epochs on it and exports the five
artifact files; API and predictor tests reuse that model. If you change the artifact format, that fixture is
the first thing to update.

### 3.4 The exception contract, in practice

Three rules cover 95 % of cases:

1. **Raise precise, catch broad.** Inner code raises `DataError`, `ModelError`, `TrainingError`,
   `CheckpointError`, `InferenceError`, `AzureError` with a `context={...}` dict of non-secret facts.
   Only three places catch: `cli._run` (log once, exit code), the API exception handlers (HTTP status + JSON +
   request id) and the MLflow run context (marks the run FAILED and re-raises).
2. **Wrap third-party errors at the module boundary.** Never let a `FileNotFoundError`, a pydantic
   `ValidationError` or a torch `RuntimeError` escape a module: `raise DataError("...", context=...) from e`.
   The `from e` keeps the original traceback.
3. **Retry only transient failures.** `@transient_retry()` retries connection errors, timeouts and HTTP
   408/429/5xx with exponential backoff and jitter, then re-raises. Never wrap validation or auth errors in it.

ruff enforces the shape (`BLE` forbids blind `except:`, `TRY` forbids log-and-re-raise and friends) and the
tests assert on exception types and `context` contents. When you add a `raise`, add a test that triggers it.

Example: adding a check that texts are not longer than N tokens.

```python
# src/textclf/data.py, inside load_split
too_long = int((df["text"].str.len() > max_chars).sum())
if too_long:
    raise DataError("Texts exceed max_chars", context={**ctx, "rows": too_long, "max_chars": max_chars})
```

```python
# tests/test_data.py
def test_load_split_rejects_long_texts(tmp_path: Path) -> None:
    _write(tmp_path, pd.DataFrame({"text": ["x" * 6000], "label": [0]}))
    with pytest.raises(DataError, match="max_chars") as info:
        load_split(tmp_path, "train", num_classes=4)
    assert info.value.context["rows"] == 1
```

### 3.5 Common changes

**Change a hyperparameter.** Edit `configs/train.yaml`. Unknown keys fail immediately with a `ConfigError`
naming the key, so typos cannot silently fall back to defaults. `--epochs N` on the CLI overrides the file.

**Add a hyperparameter.** Add the field to the right pydantic model in `config.py` (with a `Field(...)`
constraint), add it to `configs/train.yaml` (the test `test_defaults_match_committed_yaml` insists the file
equals the defaults), use it, and add a test for an out-of-range value.

**Add a model architecture.** Implement an `nn.Module` in `model.py`, register it in `_REGISTRY`, extend the
`Literal` for `ModelConfig.name`, and keep the `min_len` attribute if the model has a minimum sequence length
(the collate function pads to it).

**Add a field to the API response.** Extend `PredictionOut` in `api/schemas.py`, populate it in
`api/app.py::predict`, and assert it in `tests/test_api.py::test_predict_happy_path`.

**Look at MLflow runs locally.** Training writes to a sqlite file, `mlruns.db`:

```bash
uv run mlflow ui --backend-store-uri sqlite:///mlruns.db     # then open http://127.0.0.1:5000
```

**Resume an interrupted training.** Checkpoints are written atomically to `outputs/checkpoints/{last,best}.pt`
after every epoch; SIGTERM/SIGINT finish the current step, write `last.pt` and exit with a `TrainingError`
telling you to run `textclf train --resume-from outputs/checkpoints/last.pt`.

---

## 4. Provisioning Azure (once)

Everything in Azure is described in two Bicep files. `infra/main.bicep` creates the platform (Log Analytics,
App Insights, Storage, Key Vault, Container Registry, ML workspace, Container Apps environment and a placeholder
container app). `infra/app.bicep` describes the container app and is redeployed every time a new image exists.

### 4.1 Provision

```bash
make provision
```

That creates the resource group, lints both templates and prints a **what-if** plan (nine `Create` lines,
nothing else). Read it, then create for real:

```bash
set -a; source .env; set +a
az deployment group create -g $AZURE_RG -n main -f infra/main.bicep -p baseName=textclf -o json > docs/evidence/deploy-main.json
az deployment group show -g $AZURE_RG -n main --query properties.outputs -o table
```

Copy the `acrName` output into `.env` as `AZURE_ACR_NAME`. Then make the CLI default to this group and
workspace, and give yourself blob access on the workspace storage (uploads use your identity, not account keys):

```bash
az configure --defaults group=$AZURE_RG workspace=$AZURE_ML_WORKSPACE location=$AZURE_LOCATION
STORAGE_ID=$(az storage account show -n <storageAccountName output> -g $AZURE_RG --query id -o tsv)
MSYS_NO_PATHCONV=1 az role assignment create --assignee "$(az ad signed-in-user show --query id -o tsv)" \
  --role "Storage Blob Data Contributor" --scope "$STORAGE_ID"
```

Check: `az resource list -g $AZURE_RG -o table` shows eight resources, and the `apiFqdn` output answers a
placeholder page over HTTPS. Names contain a hash (`acrtextclf<hash>`) because registry and storage names must
be globally unique.

### 4.2 Training prerequisites

```bash
make compute   # amlcompute cluster cpu-cluster, min 0 / max 4 nodes, system-assigned identity + AcrPull on the ACR
make env       # registers the training environment; the image is built inside the ACR on first use
make data      # uploads data/ag_news as data asset ag-news:1 (19 MB)
```

The cluster costs nothing while it has zero nodes. The `AcrPull` role is what lets the cluster pull the training
image: the registry's admin user is disabled on purpose, every pull in this project uses a managed identity.

### 4.3 What it costs while idle

| Resource | Idle cost |
|---|---|
| Container Registry (Basic) | about $0.17 per day (the only fixed cost) |
| Log Analytics, App Insights | per GB ingested, effectively $0 |
| Storage (LRS) | cents per GB-month |
| Key Vault, ML workspace, Container Apps environment | $0 |
| Container app at 0 replicas, cluster at 0 nodes | $0 |

Training on one `Standard_DS3_v2` node costs a few cents per run. The load test in this repo (four replicas for
three minutes) cost less than a cent.

---

## 5. Training in the cloud

### 5.1 Smoke first, then the real run

A smoke job validates environment, inputs and outputs in a few minutes before you spend on a full run:

```bash
az ml job create -f aml/job.yaml --set inputs.epochs=1 --set display_name=textclf-smoke \
  --set command="pip install --no-deps . && textclf train --config configs/train.yaml --data-dir \${{inputs.data_dir}} --output-dir \${{outputs.model_dir}} --epochs \${{inputs.epochs}} --limit 5000" \
  --query name -o tsv
```

The full run is just the file as committed:

```bash
make submit          # az ml job create -f aml/job.yaml ... and stream the logs
```

What happens, and how long it takes the first time:

1. **Preparing** (10-12 min the first time, seconds afterwards): the training image is built in the ACR from
   `aml/Dockerfile.train`. Any later job reuses it until `pyproject.toml` or the Dockerfile change.
2. **Queued** (2-4 min): a node is being started from zero.
3. **Running**: `pip install --no-deps .` installs the code snapshot on top of the image, then `textclf train`
   runs exactly as it does locally. MLflow points at the workspace automatically (`MLFLOW_TRACKING_URI` is set
   by the job). Five epochs on 108 000 rows took 5.2 minutes.
4. **Completed**: `outputs/model_dir` holds `model/` (the five files) and `checkpoints/`.

### 5.2 Watching a job

- Studio: `az ml job show -n <job> --query services.Studio.endpoint -o tsv` prints the URL; the Metrics tab
  shows `train_loss`, `val_loss`, `val_accuracy`, `val_macro_f1` per epoch.
- CLI: `az ml job show -n <job> --query status -o tsv` (poll it if streaming misbehaves on Windows).
- Logs as files: `az ml job download -n <job> --all --download-path outputs/joblogs`. Your own output is in
  `artifacts/user_logs/std_log.txt`; image build logs are in `azureml-logs/20_image_build_log.txt`.
- The error message of a failed job is not always in `az ml job show`; the run-history API has it:

  ```bash
  MSYS_NO_PATHCONV=1 az rest --resource https://ml.azure.com --url \
    "https://<region>.api.azureml.ms/history/v1.0/subscriptions/<sub>/resourceGroups/<rg>/providers/Microsoft.MachineLearningServices/workspaces/<ws>/runs/<job>/details" \
    --query error
  ```

### 5.3 Registering the model

```bash
make register JOB=<job name>        # -> prints the new version number
az ml model list --name textclf -o table
```

Versions auto-increment. Anything registered is immutable; the CI deploy picks a version by number. To check
what CI will download:

```bash
az ml model download --name textclf --version 2 --download-path model_download
find model_download -name model.pt      # model_download/textclf/model_dir/model/model.pt
```

### 5.4 Stretch: the same pipeline from Python

`scripts/run_pipeline.py` does data asset, compute, environment, job and registration through the SDK v2, each
step wrapped in `transient_retry` and converted to `AzureError` with `context={"step", "workspace"}`:

```bash
set -a; source .env; set +a
uv run python scripts/run_pipeline.py --epochs 5
```

---

## 6. Deploying and operating the API

### 6.1 Deploying

There are two ways; both end with the same Bicep module.

**Through CI (the normal way).** Merge or push to `main`: `.github/workflows/deploy.yml` runs `ci`, logs in with
OIDC, downloads model version `MODEL_VERSION` (a repository variable), builds `textclf-api:<git sha>` in the ACR
with `az acr build`, redeploys `infra/app.bicep` with that image and smoke-tests `/ready` and `/predict`.
To deploy a specific model version without a code change, trigger it by hand:

```bash
gh workflow run deploy.yml -f model_version=2
gh run watch
```

To change the default version for future pushes: `gh variable set MODEL_VERSION --body 2`.
Pushes that only touch `docs/` or `*.md` do not rebuild the image.

**By hand (useful while iterating).** With a local model in `outputs/model`:

```bash
make deploy        # copies the model in, az acr build with tag = git sha, redeploys app.bicep
```

### 6.2 Verifying a deployment

```bash
FQDN=$(az containerapp show -n $ACA_APP -g $AZURE_RG --query properties.configuration.ingress.fqdn -o tsv)
curl -s https://$FQDN/ready
curl -s -X POST https://$FQDN/predict -H 'content-type: application/json' -H 'x-request-id: my-trace-1' \
  -d '{"texts":["Parliament approves the new border treaty"]}'
az containerapp revision list -n $ACA_APP -g $AZURE_RG -o table       # the active revision must be Healthy
az containerapp replica list  -n $ACA_APP -g $AZURE_RG -o table       # 0 when idle, up to 5 under load
az containerapp logs show     -n $ACA_APP -g $AZURE_RG --tail 50      # JSON lines
```

Expect the first request after an idle period to take 20-40 seconds (a replica starts and loads the model);
subsequent requests take well under a second.

### 6.3 Reading logs

Every response carries `x-request-id`. Every request writes one JSON line like

```json
{"ts": "...", "level": "INFO", "logger": "textclf.api.app", "msg": "POST /predict -> 200 in 41.3 ms", "request_id": "my-trace-1"}
```

and every failure writes an `ERROR` line with the same `request_id`, the exception `context` and the traceback.
The client only ever sees `{"error": "...", "detail": "...", "request_id": "..."}`; internals stay in the logs.
In the portal, Log Analytics (`log-textclf`) receives these lines; a query to find one request:

```kusto
ContainerAppConsoleLogs_CL
| where ContainerAppName_s == "textclf-api"
| where Log_s contains "my-trace-1"
| project TimeGenerated, Log_s
```

### 6.4 Scaling knobs

| Symptom | Knob | Where |
|---|---|---|
| tail latency too high under bursts | lower `concurrentRequests` (more replicas sooner) or raise `maxReplicas` | `infra/app.bicep` |
| cold starts unacceptable | `minReplicas: 1` (costs one replica 24/7) | `infra/app.bicep` |
| a single replica is slow | bigger `resources.cpu` / `memory`; `TEXTCLF_TORCH_THREADS` to match | `infra/app.bicep` env |
| many texts per request | `TEXTCLF_MAX_BATCH_SIZE` (schema caps a request at 64 texts) | env |

Load-test a change before and after (100 concurrent clients, 2 000 requests, p50/p95/p99, error rate):

```bash
uv run python scripts/loadtest.py --url https://$FQDN --requests 2000 --concurrency 100
```

and watch `az containerapp replica list` in another shell to see replicas appear.

### 6.5 Rolling back

Every image is tagged with the git SHA and every deployment is a new revision, so rolling back is redeploying
an older image:

```bash
az deployment group create -g $AZURE_RG -f infra/app.bicep -p baseName=textclf -p acrName=$AZURE_ACR_NAME \
  -p acaEnvName=$ACA_ENV -p apiImage=$AZURE_ACR_NAME.azurecr.io/textclf-api:<older sha>
```

Or re-run the deploy workflow with the previous `model_version`. Old revisions stay listed (inactive) for a
while and can be reactivated in the portal in an emergency.

---

## 7. How CI/CD works and how to repair it

### 7.1 The two workflows

- `ci.yml` (on pull requests, and called by deploy): install CPU torch, `ruff`, `mypy src`, `pytest` with the
  coverage gate, `az bicep build` on both templates. No cloud access.
- `deploy.yml` (on push to `main`, and by hand): `ci`, then a `deploy` job that needs Azure. The job is skipped
  (not failed) until the repository variable `AZURE_RG` exists, so a fresh fork does not go red.

### 7.2 Identity: no secrets anywhere

The runner authenticates with an **OIDC federated credential**: GitHub mints a short-lived token for the job,
Entra checks that the token's `subject` matches a credential on the app registration `textclf-github-oidc`,
and `azure/login` gets an access token for the service principal (Contributor on `rg-textclf` only).
GitHub stores three *ids* as secrets (`AZURE_CLIENT_ID`, `AZURE_TENANT_ID`, `AZURE_SUBSCRIPTION_ID`) and five
plain variables (`AZURE_RG`, `AZURE_ACR_NAME`, `AZURE_ML_WORKSPACE`, `ACA_ENV`, `MODEL_VERSION`).
There is no client secret to rotate or leak.

The subject GitHub presents today includes owner and repository ids, so the app registration carries two
credentials:

```
repo:Pablinki/azure-pytorch-mlops:ref:refs/heads/main
repo:Pablinki@10887474/azure-pytorch-mlops@1364655068:ref:refs/heads/main
```

If a login ever fails with `AADSTS700213: No matching federated identity record found for presented assertion
subject '...'`, copy the subject from the error message into a new credential:

```bash
cat > fedcred.json <<'EOF'
{"name": "github-main-2", "issuer": "https://token.actions.githubusercontent.com",
 "subject": "<paste the subject from the error>", "audiences": ["api://AzureADTokenExchange"]}
EOF
az ad app federated-credential create --id <APP_ID> --parameters fedcred.json
```

(On Windows pass a Windows-style path to the JSON file; `az` cannot see Git Bash's `/tmp`.)

### 7.3 Setting it up for a fork

```bash
APP_ID=$(az ad app create --display-name textclf-github-oidc --query appId -o tsv)
az ad sp create --id $APP_ID >/dev/null
SP_OID=$(az ad sp show --id $APP_ID --query id -o tsv)
# federated credential(s) as above, with your owner/repo
RG_ID=$(az group show -n $AZURE_RG --query id -o tsv)
MSYS_NO_PATHCONV=1 az role assignment create --assignee-object-id $SP_OID --assignee-principal-type ServicePrincipal --role Contributor --scope $RG_ID
gh secret set AZURE_CLIENT_ID --body "$APP_ID"
gh secret set AZURE_TENANT_ID --body "$(az account show --query tenantId -o tsv)"
gh secret set AZURE_SUBSCRIPTION_ID --body "$(az account show --query id -o tsv)"
gh variable set AZURE_RG --body "$AZURE_RG"; gh variable set AZURE_ACR_NAME --body "$AZURE_ACR_NAME"
gh variable set AZURE_ML_WORKSPACE --body "$AZURE_ML_WORKSPACE"; gh variable set ACA_ENV --body "$ACA_ENV"
gh variable set MODEL_VERSION --body 1
```

---

## 8. Day-2 operations: cost, teardown, upgrades

**Teardown.** `make destroy` deletes the resource group asynchronously (everything in it: workspace, registered
models, registry, app). The GitHub repo, the identity setup and your local files are untouched. Re-running
sections 4 and 5 recreates the platform from the Bicep and YAML files in about 15 minutes of wall clock; only
`AZURE_ACR_NAME` in `.env` and the `AZURE_ACR_NAME` GitHub variable change (new hash).

**Upgrading dependencies.** Edit the floors in `pyproject.toml`, run `uv lock && uv sync --all-extras`, then
`make check`. Two pins are deliberate: `mlflow<3` in the `train` extra (the Azure MLflow plugin is not yet
compatible with MLflow 3) and CPU-only torch in both Dockerfiles. A dependency change invalidates the training
image; the next job rebuilds it (`make env` registers a new environment version).

**Shipping a new model.** Train (section 5), register (`make register`), then either bump `MODEL_VERSION` and
push, or `gh workflow run deploy.yml -f model_version=<n>`. The API reports the package version in
`model_version`; bump `__version__` in `src/textclf/__init__.py` and `pyproject.toml` when the contract changes.

**GPU training.** Change `size` in `aml/compute.yaml` to a GPU SKU, swap the base image in
`aml/Dockerfile.train` for a CUDA one and drop the `--index-url .../cpu` from its `pip install torch`.
Mixed precision switches on automatically (`use_amp = cfg.train.amp and device.type == "cuda"`).

---

## 9. Troubleshooting: every error we hit

| Symptom | Cause | Fix |
|---|---|---|
| `az login` succeeds but `No subscriptions found` | signed in with an account that has no Azure subscription | sign in with the owning account, or grant the other account a role on the subscription |
| `MissingSubscription` on `az role assignment create` (Git Bash) | MSYS rewrote `/subscriptions/...` as a Windows path | prefix with `MSYS_NO_PATHCONV=1` |
| `az ... --parameters /tmp/x.json`: `JSON file does not exist` | `az` is a Windows program; it cannot see Git Bash's `/tmp` | use a Windows path (`C:/Users/.../x.json`) |
| `'charmap' codec can't encode` while streaming logs | Windows console encoding | the operation continues; poll with `az ... show`, download logs, or use PowerShell |
| AML job: `Failed to pull Docker image ... ACR Admin user ... or Managed Identity with AcrPull` | the cluster had no identity and the admin user is disabled | `make compute` (system-assigned identity + AcrPull); already in `aml/compute.yaml` |
| AML job: `azureml_artifacts_builder() got an unexpected keyword argument 'tracking_uri'` | MLflow 3 vs `azureml-mlflow` | `mlflow<3` in the train extra (D18) |
| AML snapshot upload is >100 MB | tool caches (`.mypy_cache` alone is ~100 MB) in the build context | they are in `.amlignore`/`.dockerignore`; keep them there |
| deploy job: `AADSTS700213 No matching federated identity record` | GitHub's OIDC subject changed shape | add a federated credential with the exact subject (section 7.2) |
| first request after deploy returns 404 for ~30 s | the previous revision still answers while the new one starts | wait; the readiness probe switches traffic when `/ready` is 200 |
| `ConfigError: Invalid config values ... extra_forbidden` | a typo or unknown key in `configs/train.yaml` | fix the key; this is the schema doing its job |
| `TrainingError: Non-finite loss` | learning rate too high or a bad batch | lower `train.lr`, check the data; the context has epoch and step |
| `TrainingError: CUDA out of memory` | batch too large for the GPU | lower `train.batch_size` or `data.max_len`, keep `amp: true` |
| API `503 model_not_ready` | the replica is still loading the model | readiness probe keeps traffic away; retry |
| API `500 inference_error` with a request id | a forward pass failed | grep the id in the logs; the context has the batch size |
| pytest `PermissionError ... pytest-of-<user>` (Windows) | a locked-down `%TEMP%` folder | already handled: pytest uses `--basetemp=.pytest_tmp` |

---

## 10. Command cheat sheet

```bash
# local
make setup | make check | make format
make prepare-data | make train-local | make serve
uv run textclf train --epochs 3            # full local run
uv run textclf evaluate --model-dir outputs/model --data-dir data/ag_news
uv run mlflow ui --backend-store-uri sqlite:///mlruns.db

# cloud, once
make provision                              # then: az deployment group create ... (section 4.1)
make compute | make env | make data

# cloud, per model
make submit                                 # or the smoke variant in section 5.1
az ml job show -n <job> --query status -o tsv
make register JOB=<job>
gh workflow run deploy.yml -f model_version=<n> && gh run watch

# operate
az containerapp revision list -n textclf-api -g rg-textclf -o table
az containerapp replica list  -n textclf-api -g rg-textclf -o table
az containerapp logs show     -n textclf-api -g rg-textclf --tail 50
uv run python scripts/loadtest.py --url https://<fqdn> --requests 2000 --concurrency 100

# stop paying
make destroy
```
