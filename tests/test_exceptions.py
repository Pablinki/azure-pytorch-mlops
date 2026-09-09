"""The exception hierarchy is a public contract: boundaries rely on it."""

from __future__ import annotations

import pytest

from textclf.exceptions import (
    AzureError,
    CheckpointError,
    ConfigError,
    DataError,
    InferenceError,
    MLProjectError,
    ModelError,
    ModelNotReadyError,
    TrainingError,
)

ALL = [
    ConfigError,
    DataError,
    ModelError,
    TrainingError,
    CheckpointError,
    InferenceError,
    ModelNotReadyError,
    AzureError,
]


@pytest.mark.parametrize("cls", ALL)
def test_every_error_derives_from_base(cls: type[MLProjectError]) -> None:
    assert issubclass(cls, MLProjectError)
    assert issubclass(cls, Exception)


def test_sub_hierarchies() -> None:
    assert issubclass(CheckpointError, TrainingError)
    assert issubclass(ModelNotReadyError, InferenceError)
    assert not issubclass(DataError, ConfigError)


@pytest.mark.parametrize(
    ("cls", "code"),
    [
        (MLProjectError, 1),
        (ConfigError, 2),
        (DataError, 3),
        (ModelError, 4),
        (TrainingError, 5),
        (CheckpointError, 5),  # inherited from TrainingError
        (InferenceError, 1),  # inherited from the base
        (ModelNotReadyError, 1),
        (AzureError, 6),
    ],
)
def test_exit_codes(cls: type[MLProjectError], code: int) -> None:
    assert cls("x").exit_code == code


def test_str_without_context_is_the_message() -> None:
    assert str(DataError("file missing")) == "file missing"


def test_str_with_context_shows_context() -> None:
    err = DataError("file missing", context={"path": "data/train.parquet", "split": "train"})
    assert err.message == "file missing"
    assert err.context == {"path": "data/train.parquet", "split": "train"}
    assert str(err) == "file missing | {'path': 'data/train.parquet', 'split': 'train'}"


def test_context_defaults_to_empty_dict() -> None:
    err = ModelError("boom")
    assert err.context == {}
    assert err.args == ("boom",)


def _read_state() -> None:
    raise KeyError("model")


def _load_checkpoint() -> None:
    """Module-boundary pattern: a third-party/stdlib error is wrapped, never leaked."""
    try:
        _read_state()
    except KeyError as e:
        raise CheckpointError("Cannot load checkpoint", context={"path": "x.pt"}) from e


def test_boundary_catches_one_type() -> None:
    """A boundary catching MLProjectError sees every precise type and keeps the chained cause."""
    with pytest.raises(MLProjectError) as info:
        _load_checkpoint()
    assert isinstance(info.value, TrainingError)
    assert isinstance(info.value.__cause__, KeyError)
