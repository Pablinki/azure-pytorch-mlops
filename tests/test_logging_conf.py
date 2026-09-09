"""JSON logging must be machine-parseable and carry request ids and exception tracebacks."""

from __future__ import annotations

import json
import logging
import sys

import pytest

from textclf.logging_conf import JsonFormatter, configure_logging


@pytest.fixture(autouse=True)
def _reset_root_logging() -> None:
    """configure_logging() attaches a handler to the captured stdout; drop it after each test."""
    yield
    logging.basicConfig(force=True, handlers=[logging.NullHandler()])


def _record(msg: str, **extra: object) -> logging.LogRecord:
    record = logging.LogRecord("textclf.test", logging.INFO, __file__, 1, msg, None, None)
    for k, v in extra.items():
        setattr(record, k, v)
    return record


def test_json_formatter_basic_fields() -> None:
    out = json.loads(JsonFormatter().format(_record("hello")))
    assert out["level"] == "INFO"
    assert out["logger"] == "textclf.test"
    assert out["msg"] == "hello"
    assert out["ts"].endswith("Z")


def test_json_formatter_includes_request_id_and_context() -> None:
    out = json.loads(
        JsonFormatter().format(_record("scored", request_id="abc-123", context={"batch": 2}))
    )
    assert out["request_id"] == "abc-123"
    assert out["context"] == {"batch": 2}


def _explode() -> None:
    raise RuntimeError("kaboom")


def test_json_formatter_includes_exception() -> None:
    record = _record("failed")
    try:
        _explode()
    except RuntimeError:
        record.exc_info = sys.exc_info()
    out = json.loads(JsonFormatter().format(record))
    assert "RuntimeError: kaboom" in out["exc"]


def test_configure_logging_json(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("LOG_FORMAT", "json")
    configure_logging("DEBUG")
    logging.getLogger("textclf.x").debug("dbg", extra={"request_id": "r1"})
    line = capsys.readouterr().out.strip().splitlines()[-1]
    assert json.loads(line)["request_id"] == "r1"


def test_configure_logging_text(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("LOG_FORMAT", "text")
    configure_logging("INFO")
    logging.getLogger("textclf.y").info("plain")
    out = capsys.readouterr().out
    assert "INFO textclf.y: plain" in out
    assert not out.lstrip().startswith("{")
