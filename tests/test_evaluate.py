"""evaluate(): metrics on a loader; run_evaluate(): per-class report on a parquet split."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import torch
from conftest import tiny_config
from torch import nn
from torch.utils.data import DataLoader

from textclf.data import build_dataloaders
from textclf.evaluate import EpochStats, evaluate, run_evaluate
from textclf.exceptions import CheckpointError, DataError, ModelError
from textclf.model import build_model


def test_evaluate_returns_stats_in_range(tiny_data_dir: Path) -> None:
    cfg = tiny_config()
    bundle = build_dataloaders(cfg, tiny_data_dir)
    model = build_model(cfg, len(bundle.vocab), len(bundle.labels))
    stats = evaluate(model, bundle.val_loader, torch.device("cpu"), nn.CrossEntropyLoss())
    assert isinstance(stats, EpochStats)
    assert stats.loss > 0
    assert 0.0 <= stats.accuracy <= 1.0
    assert 0.0 <= stats.macro_f1 <= 1.0
    assert not model.training  # evaluate switches to eval mode


def test_evaluate_empty_loader() -> None:
    model = nn.Linear(2, 2)
    empty: DataLoader[tuple[list[int], int]] = DataLoader([], batch_size=1)
    stats = evaluate(model, empty, torch.device("cpu"), nn.CrossEntropyLoss())
    assert math.isnan(stats.loss)
    assert stats.accuracy == 0.0 and stats.macro_f1 == 0.0


def test_run_evaluate_writes_report(
    tiny_model_dir: Path, tiny_data_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    metrics = run_evaluate(tiny_model_dir, tiny_data_dir, split="test")
    out = capsys.readouterr().out
    for label in ("World", "Sports", "Business", "Sci/Tech", "macro avg"):
        assert label in out
    assert metrics["split"] == "test"
    assert metrics["rows"] == 20
    assert 0.0 <= float(metrics["accuracy"]) <= 1.0  # type: ignore[arg-type]
    per_class = metrics["per_class"]
    assert isinstance(per_class, dict) and "Sports" in per_class
    written = json.loads((tiny_model_dir / "metrics_test.json").read_text())
    assert written["accuracy"] == metrics["accuracy"]


def test_run_evaluate_missing_split_is_data_error(tiny_model_dir: Path, tmp_path: Path) -> None:
    (tmp_path / "labels.json").write_text(json.dumps(["World", "Sports", "Business", "Sci/Tech"]))
    with pytest.raises(DataError, match="Split file not found"):
        run_evaluate(tiny_model_dir, tmp_path, split="test")


def test_run_evaluate_missing_model_is_model_error(tiny_data_dir: Path, tmp_path: Path) -> None:
    with pytest.raises(ModelError, match="incomplete"):
        run_evaluate(tmp_path, tiny_data_dir)


def test_run_evaluate_unwritable_metrics_is_checkpoint_error(
    tiny_model_dir: Path, tiny_data_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _fail(self: Path, *_a: object, **_k: object) -> int:
        raise OSError("read-only")

    monkeypatch.setattr(Path, "write_text", _fail)
    with pytest.raises(CheckpointError, match="Cannot write evaluation metrics"):
        run_evaluate(tiny_model_dir, tiny_data_dir)
