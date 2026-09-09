"""The CLI boundary: one catch, one log line, a stable exit code."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest
import typer
from typer.testing import CliRunner

from textclf.cli import _run, app
from textclf.exceptions import ConfigError, DataError

runner = CliRunner()


def test_help_lists_commands() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for cmd in ("prepare-data", "train", "evaluate", "serve"):
        assert cmd in result.output


def test_run_exits_with_error_exit_code(caplog: pytest.LogCaptureFixture) -> None:
    def boom() -> None:
        raise DataError("no rows", context={"split": "train"})

    with caplog.at_level(logging.ERROR, logger="textclf"), pytest.raises(typer.Exit) as info:
        _run(boom)
    assert info.value.exit_code == 3
    errors = [r for r in caplog.records if r.levelno >= logging.ERROR]
    assert len(errors) == 1  # logged exactly once, at the boundary
    assert errors[0].getMessage() == "no rows"
    assert errors[0].context == {"split": "train"}  # type: ignore[attr-defined]  # via extra=
    assert errors[0].exc_info is not None  # traceback travels with the single log line


def test_run_keyboard_interrupt_exits_130() -> None:
    def interrupted() -> None:
        raise KeyboardInterrupt

    with pytest.raises(typer.Exit) as info:
        _run(interrupted)
    assert info.value.exit_code == 130


def test_run_passes_through_success() -> None:
    called: list[bool] = []
    _run(lambda: called.append(True))
    assert called == [True]


def test_config_error_maps_to_exit_code_2(tmp_path: Path) -> None:
    from textclf.config import load_config

    with pytest.raises(typer.Exit) as info:
        _run(lambda: load_config(tmp_path / "missing.yaml"))
    assert info.value.exit_code == ConfigError.exit_code == 2
