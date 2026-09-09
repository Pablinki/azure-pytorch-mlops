"""Command line interface. This module is the CLI error boundary (section 3, rule 3).

Every command does two things only: configure logging, then hand a zero-argument callable to
`_run`, which is the single place where MLProjectError is caught, logged once and turned into a
stable exit code. Heavy imports (torch, datasets, uvicorn) are deferred inside the commands so
`textclf --help` stays instant and the serving image does not need the training extras.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Annotated

import typer

from textclf.exceptions import MLProjectError

app = typer.Typer(no_args_is_help=True, pretty_exceptions_enable=False)
log = logging.getLogger("textclf")


def _run(fn: Callable[[], object]) -> None:
    """Single error boundary for every command: log once, exit with a stable code."""
    try:
        fn()
    except MLProjectError as e:
        # One line with the traceback; `context` becomes a JSON field in container logs.
        log.exception("%s", e.message, extra={"context": e.context})
        raise typer.Exit(code=e.exit_code) from None
    except KeyboardInterrupt:
        log.warning("Interrupted by user")
        raise typer.Exit(code=130) from None


@app.command("prepare-data")
def prepare_data(
    out: Annotated[Path, typer.Option(help="Output folder for parquet + labels.json")] = Path(
        "data/ag_news"
    ),
    dataset_id: Annotated[str, typer.Option(help="Hugging Face dataset id")] = "fancyzhx/ag_news",
) -> None:
    """Download AG News and write train.parquet, test.parquet and labels.json."""
    from textclf.data import prepare_data as _prepare
    from textclf.logging_conf import configure_logging

    configure_logging()
    _run(lambda: _prepare(out, dataset_id))


@app.command()
def train(
    config: Annotated[Path, typer.Option()] = Path("configs/train.yaml"),
    data_dir: Annotated[Path, typer.Option()] = Path("data/ag_news"),
    output_dir: Annotated[Path, typer.Option()] = Path("outputs"),
    epochs: Annotated[int | None, typer.Option(help="Override config")] = None,
    limit: Annotated[int | None, typer.Option(help="Use only N training rows (smoke runs)")] = None,
    resume_from: Annotated[Path | None, typer.Option(help="Checkpoint to resume from")] = None,
) -> None:
    """Train the TextCNN and export model artifacts to OUTPUT_DIR/model."""
    from textclf.config import load_config
    from textclf.logging_conf import configure_logging
    from textclf.train import run_training

    configure_logging()
    _run(
        lambda: run_training(load_config(config), data_dir, output_dir, epochs, limit, resume_from)
    )


@app.command()
def evaluate(
    model_dir: Annotated[Path, typer.Option()] = Path("outputs/model"),
    data_dir: Annotated[Path, typer.Option()] = Path("data/ag_news"),
    split: Annotated[str, typer.Option(help="Parquet split to score")] = "test",
) -> None:
    """Score a split with the exported model and print a per-class report."""
    from textclf.evaluate import run_evaluate
    from textclf.logging_conf import configure_logging

    configure_logging()
    _run(lambda: run_evaluate(model_dir, data_dir, split))


@app.command()
def serve() -> None:
    """Start the FastAPI inference server (settings from TEXTCLF_* environment variables)."""
    import uvicorn

    from textclf.config import ServeSettings
    from textclf.logging_conf import configure_logging

    settings = ServeSettings()
    configure_logging(settings.log_level)

    def _serve() -> None:
        uvicorn.run(
            "textclf.api.app:app",
            host=settings.host,
            port=settings.port,
            workers=settings.workers,
            log_config=None,
        )

    _run(_serve)
