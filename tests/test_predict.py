"""Predictor: artifact validation, batching, and the InferenceError boundary."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
import torch

from textclf.exceptions import ConfigError, InferenceError, ModelError
from textclf.predict import Prediction, Predictor


@pytest.fixture(scope="module")
def predictor(tiny_model_dir: Path) -> Predictor:
    return Predictor.from_dir(tiny_model_dir)


def test_from_uri_requires_file_scheme() -> None:
    with pytest.raises(ConfigError, match="file://") as info:
        Predictor.from_uri("azureml://models/textclf/1")
    assert info.value.context["uri"] == "azureml://models/textclf/1"


def test_from_uri_loads(tiny_model_dir: Path) -> None:
    p = Predictor.from_uri(f"file://{tiny_model_dir.as_posix()}")
    assert p.labels == ["World", "Sports", "Business", "Sci/Tech"]
    assert p.max_len == 16
    assert p.min_len == 3  # max kernel size in the tiny config


def test_missing_artifacts_listed(tmp_path: Path, tiny_model_dir: Path) -> None:
    partial = tmp_path / "partial"
    shutil.copytree(tiny_model_dir, partial)
    (partial / "vocab.json").unlink()
    (partial / "labels.json").unlink()
    with pytest.raises(ModelError, match="incomplete") as info:
        Predictor.from_dir(partial)
    assert info.value.context["missing"] == ["vocab.json", "labels.json"]


def test_corrupt_weights_are_model_error(tmp_path: Path, tiny_model_dir: Path) -> None:
    broken = tmp_path / "broken"
    shutil.copytree(tiny_model_dir, broken)
    (broken / "model.pt").write_bytes(b"garbage")
    with pytest.raises(ModelError, match="Cannot load model artifacts") as info:
        Predictor.from_dir(broken)
    assert info.value.context["path"] == str(broken)


def test_mismatched_config_is_model_error(tmp_path: Path, tiny_model_dir: Path) -> None:
    """Weights trained with one architecture must not silently load into another."""
    other = tmp_path / "other"
    shutil.copytree(tiny_model_dir, other)
    cfg = json.loads((other / "config.json").read_text())
    cfg["model"]["num_filters"] = 99
    (other / "config.json").write_text(json.dumps(cfg))
    with pytest.raises(ModelError, match="Cannot load model artifacts") as info:
        Predictor.from_dir(other)
    assert isinstance(info.value.__cause__, RuntimeError)  # size mismatch from load_state_dict


def test_invalid_config_json_is_model_error(tmp_path: Path, tiny_model_dir: Path) -> None:
    other = tmp_path / "badcfg"
    shutil.copytree(tiny_model_dir, other)
    (other / "config.json").write_text('{"model": {"name": "bert"}}')
    with pytest.raises(ModelError, match="Cannot load model artifacts") as info:
        Predictor.from_dir(other)
    assert isinstance(info.value.__cause__, ValueError)  # pydantic ValidationError


def test_predict_shapes_and_scores(predictor: Predictor) -> None:
    preds = predictor.predict(["Shares and profit rise this quarter", "Coach praises striker"])
    assert len(preds) == 2
    for p in preds:
        assert isinstance(p, Prediction)
        assert p.label in predictor.labels
        assert set(p.scores) == set(predictor.labels)
        assert sum(p.scores.values()) == pytest.approx(1.0, abs=1e-5)
        assert p.confidence == pytest.approx(max(p.scores.values()))
        assert p.scores[p.label] == p.confidence


def test_predict_learned_the_synthetic_signal(predictor: Predictor) -> None:
    """Two epochs on 45 rows is enough to separate four disjoint vocabularies."""
    texts = [
        "election minister parliament treaty",
        "goal match coach league striker",
        "shares profit market investors revenue",
        "software chip rocket algorithm satellite",
    ]
    labels = [p.label for p in predictor.predict(texts)]
    assert labels == ["World", "Sports", "Business", "Sci/Tech"]


def test_predict_handles_empty_and_unknown_text(predictor: Predictor) -> None:
    preds = predictor.predict(["", "!!!", "zzzz qqqq"])  # no tokens / only <unk>
    assert len(preds) == 3
    assert all(0 < p.confidence <= 1 for p in preds)


def test_predict_batches(predictor: Predictor, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[int] = []
    real_forward = predictor.model.forward

    def spy(batch: torch.Tensor) -> torch.Tensor:
        calls.append(batch.shape[0])
        return real_forward(batch)

    monkeypatch.setattr(predictor.model, "forward", spy)
    out = predictor.predict(["a"] * 7, batch_size=3)
    assert len(out) == 7
    assert calls == [3, 3, 1]


def test_forward_failure_is_inference_error(
    predictor: Predictor, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(_batch: torch.Tensor) -> torch.Tensor:
        raise RuntimeError("kernel crashed")

    monkeypatch.setattr(predictor.model, "forward", boom)
    with pytest.raises(InferenceError, match="Forward pass failed") as info:
        predictor.predict(["x", "y"])
    assert info.value.context == {"batch": 2}
    assert isinstance(info.value.__cause__, RuntimeError)


def test_predict_empty_list(predictor: Predictor) -> None:
    assert predictor.predict([]) == []
