"""Metrics on a DataLoader (used inside training) and a CLI-level per-class report on a split."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from textclf.exceptions import CheckpointError

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class EpochStats:
    loss: float
    accuracy: float
    macro_f1: float


def evaluate(
    model: nn.Module,
    loader: DataLoader[tuple[list[int], int]],
    device: torch.device,
    criterion: nn.Module,
) -> EpochStats:
    """Average loss, accuracy and macro-F1 over a loader. Pure: no logging, no side effects."""
    from sklearn.metrics import accuracy_score, f1_score

    model.eval()
    total, n = 0.0, 0
    preds: list[int] = []
    targets_all: list[int] = []
    with torch.inference_mode():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            logits = model(inputs)
            total += float(criterion(logits, targets).item()) * targets.size(0)
            n += targets.size(0)
            preds.extend(logits.argmax(dim=1).tolist())
            targets_all.extend(targets.tolist())
    if n == 0:
        return EpochStats(loss=float("nan"), accuracy=0.0, macro_f1=0.0)
    return EpochStats(
        loss=total / n,
        accuracy=float(accuracy_score(targets_all, preds)),
        macro_f1=float(f1_score(targets_all, preds, average="macro", zero_division=0)),
    )


def run_evaluate(model_dir: Path, data_dir: Path, split: str = "test") -> dict[str, object]:
    """Score `<split>.parquet` with the exported model, print a per-class report, write metrics."""
    from sklearn.metrics import accuracy_score, classification_report, f1_score

    from textclf.data import load_split
    from textclf.predict import Predictor

    predictor = Predictor.from_dir(model_dir)
    df = load_split(data_dir, split, num_classes=len(predictor.labels))
    texts, y_true = df["text"].tolist(), df["label"].astype(int).tolist()
    y_pred = [predictor.labels.index(p.label) for p in predictor.predict(texts)]

    report_text = classification_report(
        y_true, y_pred, target_names=predictor.labels, digits=4, zero_division=0
    )
    report_dict = classification_report(
        y_true, y_pred, target_names=predictor.labels, output_dict=True, zero_division=0
    )
    metrics: dict[str, object] = {
        "split": split,
        "rows": len(texts),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "per_class": report_dict,
    }
    print(report_text)  # noqa: T201  # the CLI report is the product here
    out = model_dir / f"metrics_{split}.json"
    try:
        out.write_text(json.dumps(metrics, indent=2))
    except OSError as e:
        raise CheckpointError("Cannot write evaluation metrics", context={"path": str(out)}) from e
    log.info(
        "%s: accuracy=%.4f macro_f1=%.4f -> %s",
        split,
        metrics["accuracy"],
        metrics["macro_f1"],
        out,
    )
    return metrics
