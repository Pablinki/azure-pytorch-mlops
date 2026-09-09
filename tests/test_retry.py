"""transient_retry retries only transient failures and gives up after a bounded number of tries."""

from __future__ import annotations

import pytest

from textclf.retry import is_transient, transient_retry


class _HttpError(Exception):
    def __init__(self, status_code: int) -> None:
        super().__init__(f"HTTP {status_code}")
        self.status_code = status_code


class _Response:
    def __init__(self, status_code: int) -> None:
        self.status_code = status_code


class _RequestsStyleError(Exception):
    """requests/httpx/huggingface_hub expose the status on `.response`, not on the exception."""

    def __init__(self, status_code: int) -> None:
        super().__init__(f"HTTP {status_code}")
        self.response = _Response(status_code)


@pytest.mark.parametrize(
    ("exc", "expected"),
    [
        (ConnectionError("reset"), True),
        (TimeoutError("slow"), True),
        (_HttpError(429), True),
        (_HttpError(503), True),
        (_HttpError(404), False),
        (_HttpError(401), False),
        (_RequestsStyleError(502), True),
        (_RequestsStyleError(403), False),
        (ValueError("bad input"), False),
        (KeyError("k"), False),
    ],
)
def test_is_transient(exc: BaseException, expected: bool) -> None:
    assert is_transient(exc) is expected


def test_retries_transient_then_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[int] = []

    @transient_retry(attempts=4, max_wait=0.01)
    def flaky() -> str:
        calls.append(1)
        if len(calls) < 3:
            raise ConnectionError("try again")
        return "ok"

    # tenacity honours `wait`; make it instant so the suite stays fast.
    flaky.retry.wait = lambda *_: 0  # type: ignore[attr-defined]  # tenacity attaches .retry
    assert flaky() == "ok"
    assert len(calls) == 3


def test_gives_up_after_bounded_attempts_and_reraises_original() -> None:
    calls: list[int] = []

    @transient_retry(attempts=3, max_wait=0.01)
    def always_down() -> None:
        calls.append(1)
        raise _HttpError(503)

    always_down.retry.wait = lambda *_: 0  # type: ignore[attr-defined]  # tenacity attaches .retry
    # reraise=True: the caller sees the real exception, not tenacity's RetryError
    with pytest.raises(_HttpError):
        always_down()
    assert len(calls) == 3


def test_non_transient_is_not_retried() -> None:
    calls: list[int] = []

    @transient_retry(attempts=5, max_wait=0.01)
    def broken() -> None:
        calls.append(1)
        raise ValueError("programming error")

    with pytest.raises(ValueError, match="programming error"):
        broken()
    assert len(calls) == 1
