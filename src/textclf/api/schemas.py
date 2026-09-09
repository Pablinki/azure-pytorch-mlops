"""Request / response / error models. Validation here is the API's first line of defence (422)."""

from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    texts: list[Annotated[str, Field(min_length=1, max_length=5000)]] = Field(
        min_length=1, max_length=64, description="1-64 texts, each 1-5000 characters"
    )


class PredictionOut(BaseModel):
    label: str
    confidence: float
    scores: dict[str, float]


class PredictResponse(BaseModel):
    predictions: list[PredictionOut]
    model_version: str
    request_id: str


class ErrorResponse(BaseModel):
    error: str
    detail: str
    request_id: str
