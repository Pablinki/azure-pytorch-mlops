"""Training loop: failure modes, atomic checkpoints, resume, early stop, artifacts, MLflow."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import mlflow
import pytest
import torch
from conftest import tiny_config

from textclf.config import Config
from textclf.data import build_dataloaders
from textclf.exceptions import CheckpointError, TrainingError
from textclf.model import build_model
from textclf.train import ARTIFACT_FILES, Trainer, _flatten, run_training


def _trainer(data_dir: Path, out: Path, **train_overrides: object) -> Trainer:
    raw = json.loads(tiny_config().model_dump_json())
    raw["train"].update(train_overrides)
    cfg = Config.model_validate(raw)
    bundle = build_dataloaders(cfg, data_dir)
    model = build_model(cfg, len(bundle.vocab), len(bundle.labels))
    return Trainer(
        cfg, model, bundle.train_loader, bundle.val_loader, out, device=torch.device("cpu")
    )


# ---------------------------------------------------------------- failure modes


def test_non_finite_loss_raises_training_error(tiny_data_dir: Path, tmp_path: Path) -> None:
    trainer = _trainer(tiny_data_dir, tmp_path)

    class _NaNLoss(torch.nn.Module):
        def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
            return (logits.sum() * 0) + float("nan")

    trainer.criterion = _NaNLoss()
    with pytest.raises(TrainingError, match="Non-finite loss") as info:
        trainer.train_one_epoch(epoch=3)
    assert info.value.context["epoch"] == 3
    assert info.value.context["step"] == 0
    assert info.value.exit_code == 5


def test_cuda_oom_is_wrapped_with_hint(
    tiny_data_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    trainer = _trainer(tiny_data_dir, tmp_path)

    def _oom(*_a: object, **_k: object) -> float:
        raise torch.cuda.OutOfMemoryError("CUDA out of memory")

    monkeypatch.setattr(trainer, "_step", _oom)
    with pytest.raises(TrainingError, match="reduce batch_size") as info:
        trainer.train_one_epoch(epoch=0)
    assert info.value.context["batch_size"] == 8
    assert isinstance(info.value.__cause__, torch.cuda.OutOfMemoryError)


# ---------------------------------------------------------------- checkpoints


def test_checkpoint_is_atomic_and_loads(tiny_data_dir: Path, tmp_path: Path) -> None:
    trainer = _trainer(tiny_data_dir, tmp_path)
    trainer.train_one_epoch(0)
    path = tmp_path / "checkpoints" / "last.pt"
    trainer.save_checkpoint(0, path)
    assert path.is_file()
    assert list(path.parent.glob("*.tmp")) == []  # the temp file never survives
    state = torch.load(path, map_location="cpu", weights_only=True)
    assert set(state) >= {"epoch", "best_f1", "model", "optimizer", "scaler"}
    assert state["epoch"] == 0


def test_resume_continues_at_next_epoch(tiny_data_dir: Path, tmp_path: Path) -> None:
    first = _trainer(tiny_data_dir, tmp_path)
    first.train_one_epoch(0)
    first.best_f1 = 0.5
    first.save_checkpoint(0, tmp_path / "last.pt")

    second = _trainer(tiny_data_dir, tmp_path)
    second.load_checkpoint(tmp_path / "last.pt")
    assert second.start_epoch == 1
    assert second.best_f1 == 0.5
    for p1, p2 in zip(first.model.parameters(), second.model.parameters(), strict=True):
        assert torch.equal(p1, p2)


def test_save_checkpoint_error(
    tiny_data_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    trainer = _trainer(tiny_data_dir, tmp_path)

    def _fail(*_a: object, **_k: object) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(torch, "save", _fail)
    with pytest.raises(CheckpointError, match="Cannot write checkpoint") as info:
        trainer.save_checkpoint(0, tmp_path / "ck" / "last.pt")
    assert info.value.context["path"].endswith("last.pt")
    assert list((tmp_path / "ck").glob("*.tmp")) == []  # cleaned up on failure


def test_load_checkpoint_errors(tiny_data_dir: Path, tmp_path: Path) -> None:
    trainer = _trainer(tiny_data_dir, tmp_path)
    with pytest.raises(CheckpointError, match="Cannot load checkpoint") as info:
        trainer.load_checkpoint(tmp_path / "missing.pt")
    assert isinstance(info.value.__cause__, OSError)

    torch.save({"epoch": 0}, tmp_path / "partial.pt")  # missing keys
    with pytest.raises(CheckpointError, match="Cannot load checkpoint") as info:
        trainer.load_checkpoint(tmp_path / "partial.pt")
    assert isinstance(info.value.__cause__, KeyError)

    (tmp_path / "garbage.pt").write_bytes(b"not a checkpoint")
    with pytest.raises(CheckpointError, match="Cannot load checkpoint"):
        trainer.load_weights(tmp_path / "garbage.pt")


# ---------------------------------------------------------------- fit


def test_fit_stops_on_signal_after_checkpointing(tiny_data_dir: Path, tmp_path: Path) -> None:
    trainer = _trainer(tiny_data_dir, tmp_path, epochs=5)
    trainer._request_stop(15, None)  # what SIGTERM would do
    with pytest.raises(TrainingError, match="resume with --resume-from") as info:
        trainer.fit()
    assert info.value.context["epoch"] == 0
    assert Path(info.value.context["checkpoint"]).is_file()
    assert mlflow.active_run() is None  # the run context closed (marked FAILED) and re-raised


def test_fit_early_stopping(
    tiny_data_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    trainer = _trainer(tiny_data_dir, tmp_path, epochs=50, early_stopping_patience=2)
    scores = iter([0.5, 0.4, 0.3, 0.9, 0.9, 0.9])  # improves once, then plateaus

    from textclf import train as train_mod
    from textclf.evaluate import EpochStats

    monkeypatch.setattr(
        train_mod,
        "evaluate",
        lambda *_a, **_k: EpochStats(loss=1.0, accuracy=0.5, macro_f1=next(scores)),
    )
    summary = trainer.fit()
    assert summary["epochs_run"] == 3  # 0.5 best, then two non-improving epochs -> stop
    assert summary["best_val_macro_f1"] == 0.5
    assert summary["best_epoch"] == 0


def test_fit_signal_handlers_are_restored(tiny_data_dir: Path, tmp_path: Path) -> None:
    import signal

    before = signal.getsignal(signal.SIGINT)
    _trainer(tiny_data_dir, tmp_path, epochs=1).fit()
    assert signal.getsignal(signal.SIGINT) is before


# ---------------------------------------------------------------- run_training / artifacts


def test_run_training_exports_artifacts_and_logs_mlflow(tiny_model_dir: Path) -> None:
    for name in ARTIFACT_FILES:
        assert (tiny_model_dir / name).is_file(), name
    metrics = json.loads((tiny_model_dir / "metrics.json").read_text())
    assert metrics["epochs_run"] == 2
    assert 0.0 <= metrics["best_val_macro_f1"] <= 1.0
    assert metrics["wall_clock_s"] > 0
    cfg = Config.model_validate_json((tiny_model_dir / "config.json").read_text())
    assert cfg.model.kernel_sizes == [2, 3]
    assert json.loads((tiny_model_dir / "labels.json").read_text()) == [
        "World",
        "Sports",
        "Business",
        "Sci/Tech",
    ]

    runs = mlflow.search_runs(search_all_experiments=True, output_format="list")
    finished = [r for r in runs if r.info.status == "FINISHED"]
    assert finished, "the fixture's training run must be recorded as FINISHED"
    run = finished[0]
    assert run.data.params["model.kernel_sizes"] == "[2, 3]"  # dotted keys from _flatten
    assert run.data.params["device"] == "cpu"
    assert {"train_loss", "val_macro_f1", "epochs_run"} <= set(run.data.metrics)


def test_run_training_limit_and_epochs_override(tiny_data_dir: Path, tmp_path: Path) -> None:
    metrics = run_training(tiny_config(), tiny_data_dir, tmp_path, epochs=1, limit=40)
    assert metrics["epochs_run"] == 1
    assert (tmp_path / "model" / "model.pt").is_file()


def test_run_training_resume(tiny_data_dir: Path, tmp_path: Path) -> None:
    run_training(tiny_config(), tiny_data_dir, tmp_path, epochs=1)
    metrics = run_training(
        tiny_config(),
        tiny_data_dir,
        tmp_path,
        epochs=3,
        resume_from=tmp_path / "checkpoints" / "last.pt",
    )
    assert metrics["epochs_run"] == 2  # epochs 1 and 2 only


def test_flatten_produces_dotted_keys() -> None:
    assert _flatten({"a": 1, "b": {"c": 2, "d": {"e": [3]}}}) == {"a": 1, "b.c": 2, "b.d.e": [3]}
    assert _flatten({}) == {}


def test_configure_tracking_prefers_env_then_explicit_then_sqlite(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from textclf.train import LOCAL_TRACKING_URI, configure_tracking

    explicit = f"sqlite:///{(tmp_path / 'explicit.db').as_posix()}"
    mlflow.set_tracking_uri(explicit)
    assert configure_tracking() == explicit  # an explicit set_tracking_uri (tests, notebooks) wins

    monkeypatch.setenv("MLFLOW_TRACKING_URI", explicit)
    assert configure_tracking() == explicit  # AML sets this env var inside jobs

    monkeypatch.delenv("MLFLOW_TRACKING_URI")
    monkeypatch.setattr(mlflow, "is_tracking_uri_set", lambda: False)
    monkeypatch.setattr(mlflow, "get_tracking_uri", lambda: LOCAL_TRACKING_URI)
    seen: list[str] = []
    monkeypatch.setattr(mlflow, "set_tracking_uri", seen.append)
    assert configure_tracking() == LOCAL_TRACKING_URI
    assert seen == [LOCAL_TRACKING_URI]


@pytest.fixture(autouse=True)
def _no_run_leaks() -> Iterator[None]:
    """No test may leave an MLflow run active or re-point the tracking URI for the others."""
    uri = mlflow.get_tracking_uri()
    yield
    assert mlflow.active_run() is None
    mlflow.set_tracking_uri(uri)


def test_export_artifacts_attaches_to_the_run(tiny_model_dir: Path) -> None:
    metrics = json.loads((tiny_model_dir / "metrics.json").read_text())
    assert metrics["epochs_run"] == 2
    runs = [r for r in mlflow.search_runs(search_all_experiments=True, output_format="list")]
    with_artifacts = [
        r
        for r in runs
        if any(a.path == "model" for a in mlflow.MlflowClient().list_artifacts(r.info.run_id))
    ]
    assert with_artifacts, "the fixture run must carry the model/ artifact folder"


def test_export_artifacts_without_run_id_only_warns(
    tiny_data_dir: Path, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    from textclf.train import export_artifacts

    trainer = _trainer(tiny_data_dir, tmp_path)
    bundle = build_dataloaders(tiny_config(), tiny_data_dir)
    with caplog.at_level("WARNING"):
        export_artifacts(
            trainer.model, bundle.vocab, bundle.labels, tiny_config(), {}, tmp_path / "m"
        )
    assert "not attached to a run" in caplog.text
    assert (tmp_path / "m" / "model.pt").is_file()
    assert mlflow.active_run() is None
