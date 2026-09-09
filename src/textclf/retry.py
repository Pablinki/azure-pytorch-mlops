"""Bounded retry for TRANSIENT failures only (network, throttling, 5xx).

Retrying programming, validation or auth errors only delays the failure and hides the bug.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import ParamSpec, TypeVar

from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential_jitter,
)

log = logging.getLogger(__name__)

P = ParamSpec("P")
R = TypeVar("R")

TRANSIENT_STATUS = {408, 429, 500, 502, 503, 504}


def is_transient(exc: BaseException) -> bool:
    if isinstance(exc, ConnectionError | TimeoutError):
        return True
    status = getattr(exc, "status_code", None)  # azure.core HttpResponseError
    if status is None:  # requests / httpx / huggingface_hub keep it on the response object
        status = getattr(getattr(exc, "response", None), "status_code", None)
    return status in TRANSIENT_STATUS


def transient_retry(
    attempts: int = 5, max_wait: float = 30.0
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Exponential backoff with jitter: 1s, ~2s, ~4s ... capped at max_wait, then re-raise."""
    return retry(
        retry=retry_if_exception(is_transient),
        stop=stop_after_attempt(attempts),
        wait=wait_exponential_jitter(initial=1, max=max_wait),
        before_sleep=before_sleep_log(log, logging.WARNING),
        reraise=True,
    )
