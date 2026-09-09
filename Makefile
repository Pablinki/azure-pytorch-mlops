# Human interface for the project. Every target is short; the real logic lives in the package or in Azure CLI.
# Load Azure settings with `set -a; source .env; set +a` (the azure targets do it for you).
SHELL := bash
.ONESHELL:
.DEFAULT_GOAL := help

ENV := set -a; source .env; set +a;
TAG ?= $(shell git rev-parse --short HEAD 2>/dev/null || echo local)

.PHONY: help setup lint format typecheck test check prepare-data train-local serve docker-build docker-run \
        provision compute env data submit register deploy destroy

help:  ## List targets
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-14s %s\n", $$1, $$2}'

setup:  ## Create .venv with all extras and install git hooks
	uv sync --all-extras
	uv run pre-commit install

lint:  ## ruff check + format check
	uv run ruff check . && uv run ruff format --check .

format:  ## Auto-fix lint and formatting
	uv run ruff check --fix . && uv run ruff format .

typecheck:  ## mypy --strict on the package
	uv run mypy src

test:  ## pytest with coverage gate (>= 80 %)
	uv run pytest

check: lint typecheck test  ## Everything CI runs

prepare-data:  ## Download AG News → data/ag_news/{train,test}.parquet + labels.json
	uv run textclf prepare-data --out data/ag_news

train-local:  ## 1-epoch, 5000-row smoke training on CPU → outputs/model
	uv run textclf train --epochs 1 --limit 5000

serve:  ## Run the API locally against outputs/model
	uv run textclf serve

docker-build:  ## Build the serving image with the local model baked in
	rm -rf model && cp -r outputs/model model
	docker build -t textclf-api:local .

docker-run:  ## Run the local image on :8000
	docker run --rm -p 8000:8000 textclf-api:local

provision:  ## Phase 4: resource group + platform (main.bicep). Review what-if first.
	$(ENV) az group create -n $$AZURE_RG -l $$AZURE_LOCATION
	$(ENV) az bicep build -f infra/main.bicep && az bicep build -f infra/app.bicep
	$(ENV) az deployment group what-if -g $$AZURE_RG -f infra/main.bicep -p baseName=textclf

compute:  ## Phase 5: amlcompute cluster (min 0)
	$(ENV) az ml compute create -f aml/compute.yaml -g $$AZURE_RG -w $$AZURE_ML_WORKSPACE

env:  ## Phase 5: training environment image built in ACR
	$(ENV) az ml environment create -f aml/environment.yaml -g $$AZURE_RG -w $$AZURE_ML_WORKSPACE

data:  ## Phase 5: upload data/ag_news as data asset ag-news:1
	$(ENV) az ml data create -f aml/data.yaml -g $$AZURE_RG -w $$AZURE_ML_WORKSPACE

submit:  ## Phase 5: submit the training job and stream logs
	$(ENV) JOB=$$(az ml job create -f aml/job.yaml -g $$AZURE_RG -w $$AZURE_ML_WORKSPACE --query name -o tsv) \
	&& echo "JOB=$$JOB" && az ml job stream -n $$JOB -g $$AZURE_RG -w $$AZURE_ML_WORKSPACE

register:  ## Phase 5: register the job output as model textclf (JOB=<name> required)
	$(ENV) az ml model create --name textclf --type custom_model -g $$AZURE_RG -w $$AZURE_ML_WORKSPACE \
	  --path azureml://jobs/$(JOB)/outputs/model_dir --query version -o tsv

deploy:  ## Phase 7: build image in ACR (tag = git sha) and redeploy app.bicep
	$(ENV) az acr build --registry $$AZURE_ACR_NAME --image textclf-api:$(TAG) --file Dockerfile .
	$(ENV) az deployment group create -g $$AZURE_RG -f infra/app.bicep -p baseName=textclf \
	  -p acrName=$$AZURE_ACR_NAME -p acaEnvName=$$ACA_ENV -p apiImage=$$AZURE_ACR_NAME.azurecr.io/textclf-api:$(TAG)

destroy:  ## Delete the whole resource group (stops all billing)
	$(ENV) az group delete -n $$AZURE_RG --yes --no-wait
