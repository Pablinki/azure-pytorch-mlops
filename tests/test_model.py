"""TextCNN shape contract and its one explicit failure mode."""

from __future__ import annotations

import pytest
import torch

from textclf.config import Config
from textclf.exceptions import ModelError
from textclf.model import TextCNN, build_model, count_parameters


def test_forward_shape() -> None:
    model = TextCNN(vocab_size=50, num_classes=4, embed_dim=8, num_filters=3, kernel_sizes=(2, 3))
    logits = model(torch.randint(0, 50, (5, 7)))
    assert logits.shape == (5, 4)
    assert model.min_len == 3


def test_sequence_shorter_than_kernel_is_model_error() -> None:
    model = TextCNN(vocab_size=50, num_classes=4, kernel_sizes=(3, 4, 5))
    with pytest.raises(ModelError) as info:
        model(torch.zeros(2, 4, dtype=torch.long))
    assert info.value.context == {"len": 4, "min_len": 5}


def test_padding_row_is_zero_embedding() -> None:
    model = TextCNN(vocab_size=10, num_classes=2, embed_dim=4, kernel_sizes=(1,))
    assert torch.all(model.embedding.weight[0] == 0)


def test_empty_kernel_sizes_rejected() -> None:
    with pytest.raises(ModelError, match="kernel size"):
        TextCNN(vocab_size=10, num_classes=2, kernel_sizes=())


def test_build_model_from_config() -> None:
    cfg = Config.model_validate(
        {"model": {"embed_dim": 16, "num_filters": 4, "kernel_sizes": [2, 3, 4], "dropout": 0.1}}
    )
    model = build_model(cfg, vocab_size=100, num_classes=4)
    assert isinstance(model, TextCNN)
    assert len(model.convs) == 3
    assert model.fc.in_features == 4 * 3
    assert model.fc.out_features == 4
    assert model.dropout.p == pytest.approx(0.1)
    # embedding 100*16 + 3 convs (16*k*4 + 4) + fc (12*4 + 4)
    expected = 100 * 16 + sum(16 * k * 4 + 4 for k in (2, 3, 4)) + 12 * 4 + 4
    assert count_parameters(model) == expected


def test_build_model_unknown_name_is_model_error() -> None:
    cfg = Config()
    cfg.model.__dict__["name"] = "bert"  # bypass the Literal on purpose
    with pytest.raises(ModelError, match="Unknown model") as info:
        build_model(cfg, vocab_size=10, num_classes=2)
    assert info.value.context["known"] == ["textcnn"]


def test_eval_mode_is_deterministic() -> None:
    model = TextCNN(vocab_size=50, num_classes=4, dropout=0.5, kernel_sizes=(2,))
    x = torch.randint(0, 50, (3, 6))
    model.eval()
    with torch.inference_mode():
        assert torch.equal(model(x), model(x))
