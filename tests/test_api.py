"""API boundary: status mapping, request ids, validation, readiness and fail-fast startup."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from textclf import __version__
from textclf.api.app import app
from textclf.exceptions import InferenceError, ModelError
from textclf.predict import Predictor


@pytest.fixture
def client(tiny_model_dir: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    monkeypatch.setenv("TEXTCLF_MODEL_URI", f"file://{tiny_model_dir.as_posix()}")
    monkeypatch.setenv("TEXTCLF_MAX_BATCH_SIZE", "2")
    with TestClient(app) as c:  # `with` runs the lifespan, i.e. loads the model
        yield c


def test_health_and_ready(client: TestClient) -> None:
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "version": __version__}
    assert client.get("/ready").json() == {"status": "ready"}


def test_predict_happy_path(client: TestClient) -> None:
    r = client.post(
        "/predict",
        json={"texts": ["shares profit market investors", "goal match coach league striker"]},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["model_version"] == __version__
    assert len(body["predictions"]) == 2
    assert body["predictions"][0]["label"] == "Business"
    assert body["predictions"][1]["label"] == "Sports"
    for p in body["predictions"]:
        assert set(p["scores"]) == {"World", "Sports", "Business", "Sci/Tech"}
        assert 0 < p["confidence"] <= 1
    assert body["request_id"] == r.headers["x-request-id"]


def test_predict_uses_configured_batch_size(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[int] = []
    predictor: Predictor = app.state.predictor
    real = predictor.predict

    def spy(texts: list[str], batch_size: int = 64) -> object:
        seen.append(batch_size)
        return real(texts, batch_size)

    monkeypatch.setattr(predictor, "predict", spy)
    assert client.post("/predict", json={"texts": ["a", "b", "c"]}).status_code == 200
    assert seen == [2]  # TEXTCLF_MAX_BATCH_SIZE from the fixture


@pytest.mark.parametrize(
    "payload",
    [
        {"texts": []},  # empty list
        {"texts": ["x"] * 65},  # too many
        {"texts": [""]},  # empty string
        {"texts": ["x" * 5001]},  # too long
        {"texts": ["ok"], "lang": "en"},  # unknown field
        {"text": "ok"},  # wrong key
        {"texts": "not a list"},
    ],
)
def test_validation_errors_are_422(client: TestClient, payload: dict[str, object]) -> None:
    r = client.post("/predict", json=payload)
    assert r.status_code == 422
    assert "x-request-id" in r.headers


def test_not_ready_is_503(client: TestClient) -> None:
    saved = app.state.predictor
    try:
        app.state.predictor = None
        r = client.get("/ready")
        assert r.status_code == 503
        assert r.json()["error"] == "model_not_ready"
        assert r.json()["request_id"] == r.headers["x-request-id"]
        assert client.post("/predict", json={"texts": ["x"]}).status_code == 503
    finally:
        app.state.predictor = saved


def test_inference_error_is_500_with_request_id(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    def boom(*_a: object, **_k: object) -> object:
        raise InferenceError("Forward pass failed", context={"batch": 1})

    monkeypatch.setattr(app.state.predictor, "predict", boom)
    with caplog.at_level("ERROR"):
        r = client.post("/predict", json={"texts": ["x"]}, headers={"x-request-id": "req-42"})
    assert r.status_code == 500
    assert r.json() == {
        "error": "inference_error",
        "detail": "Prediction failed; see request_id in the logs",
        "request_id": "req-42",
    }
    assert r.headers["x-request-id"] == "req-42"
    assert "req-42" in caplog.text  # the log line carries the id the client got
    assert "Forward pass failed" in caplog.text  # the detail is in the logs, not in the response


def test_other_domain_errors_are_500_with_type_name(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(*_a: object, **_k: object) -> object:
        raise ModelError("weights vanished")

    monkeypatch.setattr(app.state.predictor, "predict", boom)
    r = client.post("/predict", json={"texts": ["x"]})
    assert r.status_code == 500
    assert r.json()["error"] == "ModelError"
    assert "weights vanished" not in r.text  # internals never leak to the client


def test_request_id_echoed_or_generated(client: TestClient) -> None:
    echoed = client.get("/health", headers={"x-request-id": "abc-123"})
    assert echoed.headers["x-request-id"] == "abc-123"
    generated = client.get("/health")
    assert len(generated.headers["x-request-id"]) == 36  # uuid4


def test_startup_fails_fast_when_model_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TEXTCLF_MODEL_URI", f"file://{tmp_path.as_posix()}")
    with pytest.raises(ModelError, match="incomplete"), TestClient(app):
        pass
