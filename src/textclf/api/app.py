"""FastAPI service. This module is the API error boundary (section 3, rule 3).

Domain errors are mapped to HTTP status + JSON body + request id in exactly one place (the
exception handlers below). Request validation is pydantic's job (422). Everything else is a 500 with
the request id, and the traceback lives in the logs under that id.
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse

from textclf import __version__
from textclf.api.schemas import ErrorResponse, PredictionOut, PredictRequest, PredictResponse
from textclf.config import ServeSettings
from textclf.exceptions import InferenceError, MLProjectError, ModelNotReadyError
from textclf.logging_conf import configure_logging
from textclf.predict import Predictor

log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = ServeSettings()
    configure_logging(settings.log_level)
    torch.set_num_threads(settings.torch_threads)
    app.state.settings, app.state.predictor = settings, None
    # Fail fast: if the model cannot load, the process exits, the platform restarts the replica and
    # the readiness probe keeps traffic away. A replica that answers /health but cannot predict is
    # worse than no replica.
    # ModelError / ConfigError propagate on purpose.
    app.state.predictor = Predictor.from_uri(settings.model_uri)
    log.info("model loaded from %s (torch threads=%d)", settings.model_uri, settings.torch_threads)
    yield
    app.state.predictor = None


app = FastAPI(title="textclf-api", version=__version__, lifespan=lifespan)


@app.middleware("http")
async def request_id_middleware(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    request.state.request_id = request.headers.get("x-request-id", str(uuid.uuid4()))
    response = await call_next(request)
    response.headers["x-request-id"] = request.state.request_id
    return response


def _request_id(request: Request) -> str:
    return str(getattr(request.state, "request_id", "unknown"))


def _error(request: Request, status: int, error: str, detail: str) -> JSONResponse:
    body = ErrorResponse(error=error, detail=detail, request_id=_request_id(request))
    return JSONResponse(status_code=status, content=body.model_dump())


@app.exception_handler(ModelNotReadyError)
async def _not_ready(request: Request, exc: ModelNotReadyError) -> JSONResponse:
    return _error(request, 503, "model_not_ready", str(exc))


@app.exception_handler(InferenceError)
async def _inference(request: Request, exc: InferenceError) -> JSONResponse:
    log.error(
        "inference failed request_id=%s %s",
        _request_id(request),
        exc,
        exc_info=exc,
        extra={"request_id": _request_id(request), "context": exc.context},
    )
    return _error(request, 500, "inference_error", "Prediction failed; see request_id in the logs")


@app.exception_handler(MLProjectError)
async def _domain(request: Request, exc: MLProjectError) -> JSONResponse:
    log.error(
        "domain error request_id=%s %s",
        _request_id(request),
        exc,
        exc_info=exc,
        extra={"request_id": _request_id(request), "context": exc.context},
    )
    return _error(request, 500, type(exc).__name__, "Internal error; see request_id in the logs")


def _predictor(request: Request) -> Predictor:
    predictor: Predictor | None = request.app.state.predictor
    if predictor is None:
        raise ModelNotReadyError("Model is still loading")
    return predictor


@app.get("/health")
async def health() -> dict[str, str]:  # liveness: the process is up
    return {"status": "ok", "version": __version__}


@app.get("/ready")
async def ready(request: Request) -> dict[str, str]:  # readiness: the model is loaded
    _predictor(request)
    return {"status": "ready"}


@app.post(
    "/predict",
    response_model=PredictResponse,
    responses={500: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
)
def predict(request: Request, body: PredictRequest) -> PredictResponse:
    # Plain `def`: CPU-bound work runs in the threadpool, so the event loop keeps answering probes.
    settings: ServeSettings = request.app.state.settings
    preds = _predictor(request).predict(body.texts, batch_size=settings.max_batch_size)
    return PredictResponse(
        predictions=[
            PredictionOut(label=p.label, confidence=p.confidence, scores=p.scores) for p in preds
        ],
        model_version=__version__,
        request_id=_request_id(request),
    )
