"""Training loop. Only _step/evaluate touch tensors; everything else is plumbing.

Failure modes are explicit (section 3, rule 6):
- non-finite loss            -> TrainingError with epoch/step/loss
- CUDA out of memory         -> TrainingError with an actionable hint
- checkpoint read/write      -> CheckpointError (writes are atomic: temp file + os.replace)
- SIGTERM / SIGINT           -> finish the step, checkpoint, raise TrainingError; --resume-from
"""

from __future__ import annotations

import json
import logging
import os
import random
import signal
import tempfile
import time
from pathlib import Path
from types import FrameType
from typing import Any

import mlflow
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from textclf.config import Config
from textclf.data import Vocab, build_dataloaders
from textclf.evaluate import evaluate
from textclf.exceptions import CheckpointError, TrainingError
from textclf.model import TORCH_LOAD_ERRORS, build_model, count_parameters

log = logging.getLogger(__name__)

ARTIFACT_FILES = ("model.pt", "vocab.json", "labels.json", "config.json", "metrics.json")
LOCAL_TRACKING_URI = "sqlite:///mlruns.db"


def select_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def configure_tracking() -> str:
    """Pick the MLflow backend once: env (AML) wins, then an explicit set_tracking_uri, else sqlite.

    MLflow 3 refuses the legacy `./mlruns` file store, so the local default is a sqlite file in the
    working directory. Azure ML jobs export MLFLOW_TRACKING_URI (azureml://...) automatically.
    """
    explicitly_set = bool(mlflow.is_tracking_uri_set())  # type: ignore[no-untyped-call]  # mlflow stub gap
    if not os.getenv("MLFLOW_TRACKING_URI") and not explicitly_set:
        mlflow.set_tracking_uri(LOCAL_TRACKING_URI)
    uri = mlflow.get_tracking_uri()
    log.info("mlflow tracking uri: %s", uri)
    return uri


class Trainer:
    def __init__(
        self,
        cfg: Config,
        model: nn.Module,
        train_loader: DataLoader[tuple[list[int], int]],
        val_loader: DataLoader[tuple[list[int], int]],
        output_dir: Path,
        device: torch.device | None = None,
    ) -> None:
        self.cfg, self.model = cfg, model
        self.train_loader, self.val_loader, self.output_dir = train_loader, val_loader, output_dir
        self.device = device or select_device()
        self.model.to(self.device)
        self.use_amp = cfg.train.amp and self.device.type == "cuda"  # AMP is a no-op on CPU
        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay
        )
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        self.criterion: nn.Module = nn.CrossEntropyLoss()
        self.start_epoch, self.best_f1, self.best_epoch = 0, -1.0, -1
        self.best_accuracy = 0.0
        self.run_id: str | None = None  # set by fit(); export attaches artifacts to it explicitly
        self._stop = False

    # ------------------------------------------------------------------ signals
    def _request_stop(self, signum: int, _frame: FrameType | None) -> None:
        log.warning("Signal %s received: finishing the current step, then checkpointing", signum)
        self._stop = True

    def _install_signal_handlers(self) -> dict[int, Any]:
        """AML preemption / `az ml job cancel` send SIGTERM; Ctrl-C sends SIGINT."""
        previous: dict[int, Any] = {}
        for sig in (signal.SIGTERM, signal.SIGINT):
            try:
                previous[sig] = signal.signal(sig, self._request_stop)
            except ValueError:  # not the main thread (e.g. a test runner worker); keep defaults
                log.debug("Cannot install handler for signal %s outside the main thread", sig)
        return previous

    @staticmethod
    def _restore_signal_handlers(previous: dict[int, Any]) -> None:
        for sig, handler in previous.items():
            signal.signal(sig, handler)

    # ------------------------------------------------------------------ one step / one epoch
    def _step(self, inputs: torch.Tensor, targets: torch.Tensor, *, epoch: int, step: int) -> float:
        with torch.autocast(
            device_type=self.device.type, dtype=torch.float16, enabled=self.use_amp
        ):
            loss = self.criterion(self.model(inputs), targets)
        if not torch.isfinite(loss):
            raise TrainingError(
                "Non-finite loss: lower lr or inspect the batch",
                context={"epoch": epoch, "step": step, "loss": float(loss.item())},
            )
        self.optimizer.zero_grad(set_to_none=True)
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)
        nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.train.max_grad_norm)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        return float(loss.item())

    def train_one_epoch(self, epoch: int) -> float:
        self.model.train()
        total, n = 0.0, 0
        for step, (inputs, targets) in enumerate(self.train_loader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            try:
                loss = self._step(inputs, targets, epoch=epoch, step=step)
            except torch.cuda.OutOfMemoryError as e:
                raise TrainingError(
                    "CUDA out of memory: reduce batch_size or max_len, or enable amp",
                    context={
                        "epoch": epoch,
                        "step": step,
                        "batch_size": self.cfg.train.batch_size,
                    },
                ) from e
            total, n = total + loss * targets.size(0), n + targets.size(0)
            if self._stop:
                break
        return total / max(n, 1)

    # ------------------------------------------------------------------ checkpoints
    def save_checkpoint(self, epoch: int, path: Path) -> None:
        state = {
            "epoch": epoch,
            "best_f1": self.best_f1,
            "best_epoch": self.best_epoch,
            "best_accuracy": self.best_accuracy,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scaler": self.scaler.state_dict(),
        }
        tmp: str | None = None
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
            with os.fdopen(fd, "wb") as f:
                torch.save(state, f)
            os.replace(tmp, path)  # atomic: a reader never sees a half-written file
        except OSError as e:
            if tmp is not None and os.path.exists(tmp):
                os.unlink(tmp)
            raise CheckpointError("Cannot write checkpoint", context={"path": str(path)}) from e

    def load_checkpoint(self, path: Path) -> None:
        try:
            state = torch.load(path, map_location=self.device, weights_only=True)
            self.model.load_state_dict(state["model"])
            self.optimizer.load_state_dict(state["optimizer"])
            self.scaler.load_state_dict(state["scaler"])
            self.start_epoch, self.best_f1 = state["epoch"] + 1, state["best_f1"]
            self.best_epoch = state.get("best_epoch", -1)
            self.best_accuracy = state.get("best_accuracy", 0.0)
        except TORCH_LOAD_ERRORS as e:
            raise CheckpointError("Cannot load checkpoint", context={"path": str(path)}) from e
        log.info(
            "resumed from %s: next epoch=%d best_f1=%.4f", path, self.start_epoch, self.best_f1
        )

    def load_weights(self, path: Path) -> None:
        """Load only the model weights (used to restore best.pt before export)."""
        try:
            state = torch.load(path, map_location=self.device, weights_only=True)
            self.model.load_state_dict(state["model"])
        except TORCH_LOAD_ERRORS as e:
            raise CheckpointError("Cannot load checkpoint", context={"path": str(path)}) from e

    # ------------------------------------------------------------------ fit
    def fit(self) -> dict[str, float]:
        patience_left = self.cfg.train.early_stopping_patience
        ckpt_dir = self.output_dir / "checkpoints"
        started = time.perf_counter()
        epochs_run = 0
        previous = self._install_signal_handlers()
        try:
            configure_tracking()
            # The context manager marks the run FAILED and re-raises on any exception (boundary #3).
            # Inside an AML job MLFLOW_RUN_ID is set, so start_run() attaches to the job's own run.
            with mlflow.start_run() as run:
                self.run_id = run.info.run_id
                mlflow.log_params(_flatten(self.cfg.model_dump()))
                mlflow.log_param("device", self.device.type)
                mlflow.log_param("parameters", count_parameters(self.model))
                for epoch in range(self.start_epoch, self.cfg.train.epochs):
                    train_loss = self.train_one_epoch(epoch)
                    stats = evaluate(self.model, self.val_loader, self.device, self.criterion)
                    epochs_run += 1
                    mlflow.log_metrics(
                        {
                            "train_loss": train_loss,
                            "val_loss": stats.loss,
                            "val_accuracy": stats.accuracy,
                            "val_macro_f1": stats.macro_f1,
                        },
                        step=epoch,
                    )
                    log.info(
                        "epoch=%d train_loss=%.4f val_loss=%.4f val_acc=%.4f val_f1=%.4f",
                        epoch,
                        train_loss,
                        stats.loss,
                        stats.accuracy,
                        stats.macro_f1,
                    )
                    if stats.macro_f1 > self.best_f1:
                        self.best_f1, self.best_epoch = stats.macro_f1, epoch
                        self.best_accuracy = stats.accuracy
                        patience_left = self.cfg.train.early_stopping_patience
                        self.save_checkpoint(epoch, ckpt_dir / "best.pt")
                    else:
                        patience_left -= 1
                    self.save_checkpoint(epoch, ckpt_dir / "last.pt")
                    if self._stop:
                        raise TrainingError(
                            "Interrupted by signal; resume with --resume-from",
                            context={"epoch": epoch, "checkpoint": str(ckpt_dir / "last.pt")},
                        )
                    if patience_left == 0:
                        log.info("Early stopping at epoch %d", epoch)
                        break
                summary = {
                    "best_val_macro_f1": self.best_f1,
                    "best_val_accuracy": self.best_accuracy,
                    "best_epoch": float(self.best_epoch),
                    "epochs_run": float(epochs_run),
                    "wall_clock_s": time.perf_counter() - started,
                }
                mlflow.log_metrics(summary)
                return summary
        finally:
            self._restore_signal_handlers(previous)


def _flatten(d: dict[str, object], prefix: str = "") -> dict[str, object]:
    out: dict[str, object] = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        out.update(_flatten(v, f"{key}.") if isinstance(v, dict) else {key: v})
    return out


def export_artifacts(
    model: nn.Module,
    vocab: Vocab,
    labels: list[str],
    cfg: Config,
    metrics: dict[str, float],
    out_dir: Path,
    run_id: str | None = None,
) -> None:
    """Write the five files the Predictor needs and attach them to the MLflow run `run_id`.

    The run has already ended when this is called, so the artifacts are logged through the client
    API by id: the fluent `mlflow.log_artifacts` would silently open a NEW run and leave it active.
    """
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), out_dir / "model.pt")
        vocab.save(out_dir / "vocab.json")
        (out_dir / "labels.json").write_text(json.dumps(labels, indent=2))
        (out_dir / "config.json").write_text(cfg.model_dump_json(indent=2))
        (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    except OSError as e:
        raise CheckpointError(
            "Cannot export model artifacts", context={"path": str(out_dir)}
        ) from e
    if run_id is None:
        log.warning("No MLflow run id: artifacts written to %s but not attached to a run", out_dir)
    else:
        mlflow.MlflowClient().log_artifacts(run_id, str(out_dir), artifact_path="model")
    log.info("exported %s to %s", ", ".join(ARTIFACT_FILES), out_dir)


def run_training(
    cfg: Config,
    data_dir: Path,
    output_dir: Path,
    epochs: int | None = None,
    limit: int | None = None,
    resume_from: Path | None = None,
) -> dict[str, float]:
    """Orchestration: seed -> data -> model -> Trainer.fit -> reload best -> export artifacts."""
    set_seed(cfg.train.seed)
    if epochs is not None:
        cfg = cfg.model_copy(update={"train": cfg.train.model_copy(update={"epochs": epochs})})
    bundle = build_dataloaders(cfg, data_dir, limit)
    model = build_model(cfg, vocab_size=len(bundle.vocab), num_classes=len(bundle.labels))
    log.info("model=%s parameters=%d", cfg.model.name, count_parameters(model))
    trainer = Trainer(cfg, model, bundle.train_loader, bundle.val_loader, output_dir)
    if resume_from is not None:
        trainer.load_checkpoint(resume_from)
    metrics = trainer.fit()
    trainer.load_weights(output_dir / "checkpoints" / "best.pt")
    export_artifacts(
        model, bundle.vocab, bundle.labels, cfg, metrics, output_dir / "model", trainer.run_id
    )
    return metrics
