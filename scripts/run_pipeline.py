"""Stretch: the Phase 5 sequence with the Azure ML SDK v2, under the textclf exception contract.

    data asset -> compute -> environment -> command job (streamed) -> registered model

Every step is retried on transient failures only and converted into AzureError with
context={"step", "workspace"}; auth problems get an actionable message. Equivalent CLI: `make data
compute env submit register`.

Usage:
    set -a; source .env; set +a
    uv run python scripts/run_pipeline.py --epochs 5
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable
from pathlib import Path
from typing import Annotated, TypeVar

import typer
from azure.ai.ml import Input, MLClient, Output, command
from azure.ai.ml.entities import AmlCompute, BuildContext, Data, Environment, Model
from azure.core.exceptions import AzureError as AzureSdkError
from azure.core.exceptions import ClientAuthenticationError
from azure.identity import DefaultAzureCredential

from textclf.exceptions import AzureError, ConfigError
from textclf.logging_conf import configure_logging
from textclf.retry import transient_retry

log = logging.getLogger("textclf.pipeline")
ROOT = Path(__file__).resolve().parents[1]
T = TypeVar("T")


def _env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise ConfigError(f"{name} is not set (load .env first)", context={"variable": name})
    return value


def _step(name: str, workspace: str, fn: Callable[[], T]) -> T:
    """Run one Azure operation with bounded retries and map SDK errors to AzureError."""
    log.info("step=%s workspace=%s", name, workspace)
    try:
        result: T = transient_retry()(fn)()
    except ClientAuthenticationError as e:
        raise AzureError(
            "Azure authentication failed: run `az login` or assign the missing role",
            context={"step": name, "workspace": workspace},
        ) from e
    except AzureSdkError as e:  # HttpResponseError, ServiceRequestError, MlException, ...
        raise AzureError(f"{name} failed", context={"step": name, "workspace": workspace}) from e
    except OSError as e:  # local upload paths (data asset, build context)
        raise AzureError(f"{name} failed reading local files", context={"step": name}) from e
    else:
        return result


def _client() -> MLClient:
    return MLClient(
        DefaultAzureCredential(),
        subscription_id=_env("AZURE_SUBSCRIPTION_ID"),
        resource_group_name=_env("AZURE_RG"),
        workspace_name=_env("AZURE_ML_WORKSPACE"),
    )


def run_pipeline(epochs: int, data_dir: Path, data_version: str, register_as: str) -> str:
    ws = _env("AZURE_ML_WORKSPACE")
    ml = _client()

    data = _step(
        "data",
        ws,
        lambda: ml.data.create_or_update(
            Data(name="ag-news", version=data_version, type="uri_folder", path=str(data_dir))
        ),
    )
    _step(
        "compute",
        ws,
        lambda: ml.compute.begin_create_or_update(
            AmlCompute(
                name="cpu-cluster",
                size="Standard_DS3_v2",
                min_instances=0,
                max_instances=4,
                idle_time_before_scale_down=120,
            )
        ).result(),
    )
    env = _step(
        "environment",
        ws,
        lambda: ml.environments.create_or_update(
            Environment(
                name="textclf-train",
                build=BuildContext(path=str(ROOT), dockerfile_path="aml/Dockerfile.train"),
            )
        ),
    )
    job = command(
        code=str(ROOT),
        command=(
            "pip install --no-deps . && textclf train --config configs/train.yaml "
            "--data-dir ${{inputs.data_dir}} --output-dir ${{outputs.model_dir}} "
            "--epochs ${{inputs.epochs}}"
        ),
        inputs={
            "data_dir": Input(type="uri_folder", path=f"{data.name}:{data.version}"),
            "epochs": epochs,
        },
        outputs={"model_dir": Output(type="uri_folder")},
        environment=f"{env.name}:{env.version}",
        compute="cpu-cluster",
        experiment_name="textclf",
        display_name="textclf-textcnn-sdk",
        tags={"project": "textclf", "framework": "pytorch", "submitted_by": "sdk-v2"},
    )
    submitted = _step("job", ws, lambda: ml.jobs.create_or_update(job))
    log.info("job %s submitted: %s", submitted.name, submitted.studio_url)
    _step("stream", ws, lambda: ml.jobs.stream(submitted.name))

    model = _step(
        "register",
        ws,
        lambda: ml.models.create_or_update(
            Model(
                name=register_as,
                path=f"azureml://jobs/{submitted.name}/outputs/model_dir",
                type="custom_model",
                description="TextCNN on AG News (SDK v2 pipeline)",
            )
        ),
    )
    log.info("registered %s:%s", model.name, model.version)
    return str(model.version)


def main(
    epochs: Annotated[int, typer.Option(min=1)] = 5,
    data_dir: Annotated[Path, typer.Option()] = Path("data/ag_news"),
    data_version: Annotated[str, typer.Option()] = "1",
    register_as: Annotated[str, typer.Option()] = "textclf",
) -> None:
    configure_logging()
    try:
        version = run_pipeline(epochs, data_dir, data_version, register_as)
    except AzureError as e:  # script-level boundary, same contract as the CLI
        log.exception("%s", e.message, extra={"context": e.context})
        raise typer.Exit(code=e.exit_code) from None
    except ConfigError as e:
        log.exception("%s", e.message, extra={"context": e.context})
        raise typer.Exit(code=e.exit_code) from None
    print(f"MODEL_VERSION={version}")  # noqa: T201


if __name__ == "__main__":
    typer.run(main)
