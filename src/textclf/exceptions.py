"""Domain exception hierarchy for textclf.

Design:
- Every error raised by this package derives from MLProjectError, so a boundary
  (CLI, API) can catch ONE type while inner code raises PRECISE types.
- Third-party exceptions never escape a module: wrap them with `raise X(...) from e`.
- `context` carries structured, non-secret facts that make logs actionable.
"""

from __future__ import annotations

from typing import Any


class MLProjectError(Exception):
    """Base class for all textclf errors."""

    exit_code: int = 1  # consumed by the CLI boundary

    def __init__(self, message: str, *, context: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.context = context or {}

    def __str__(self) -> str:
        return f"{self.message} | {self.context}" if self.context else self.message


class ConfigError(MLProjectError):
    """Missing or invalid configuration / CLI arguments."""

    exit_code = 2


class DataError(MLProjectError):
    """Dataset missing, corrupt, empty, or with an unexpected schema."""

    exit_code = 3


class ModelError(MLProjectError):
    """Model construction or artifact loading failed."""

    exit_code = 4


class TrainingError(MLProjectError):
    """Training diverged (non-finite loss), ran out of memory, or was interrupted."""

    exit_code = 5


class CheckpointError(TrainingError):
    """A checkpoint could not be written or read."""


class InferenceError(MLProjectError):
    """A valid request could not be scored."""


class ModelNotReadyError(InferenceError):
    """The service has not finished loading the model."""


class AzureError(MLProjectError):
    """An Azure operation failed after bounded retries."""

    exit_code = 6
