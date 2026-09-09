"""Model loading and batched inference. No web framework here on purpose."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch

from textclf.config import Config
from textclf.data import Vocab, collate_ids
from textclf.exceptions import ConfigError, InferenceError, ModelError
from textclf.model import TORCH_LOAD_ERRORS, build_model


@dataclass(frozen=True)
class Prediction:
    label: str
    confidence: float
    scores: dict[str, float]


class Predictor:
    REQUIRED = ("model.pt", "vocab.json", "labels.json", "config.json")

    def __init__(
        self, model: torch.nn.Module, vocab: Vocab, labels: list[str], max_len: int, min_len: int
    ) -> None:
        self.model, self.vocab, self.labels = model, vocab, labels
        self.max_len, self.min_len = max_len, min_len

    @classmethod
    def from_uri(cls, uri: str) -> Predictor:
        if not uri.startswith("file://"):
            raise ConfigError("Only file:// model URIs are supported in v1", context={"uri": uri})
        return cls.from_dir(Path(uri.removeprefix("file://")))

    @classmethod
    def from_dir(cls, path: Path) -> Predictor:
        missing = [f for f in cls.REQUIRED if not (path / f).exists()]
        if missing:
            raise ModelError(
                "Model artifacts incomplete", context={"path": str(path), "missing": missing}
            )
        try:
            cfg = Config.model_validate_json((path / "config.json").read_text())
            labels: list[str] = json.loads((path / "labels.json").read_text())
            vocab = Vocab.load(path / "vocab.json")
            model = build_model(cfg, vocab_size=len(vocab), num_classes=len(labels))
            state = torch.load(path / "model.pt", map_location="cpu", weights_only=True)
            model.load_state_dict(state)
        except TORCH_LOAD_ERRORS as e:  # ValueError also covers json + pydantic
            raise ModelError("Cannot load model artifacts", context={"path": str(path)}) from e
        model.eval()
        predictor = cls(model, vocab, labels, cfg.data.max_len, getattr(model, "min_len", 1))
        predictor.predict(["warmup"])  # allocate buffers before the first real request
        return predictor

    def predict(self, texts: list[str], batch_size: int = 64) -> list[Prediction]:
        out: list[Prediction] = []
        for i in range(0, len(texts), batch_size):
            chunk = texts[i : i + batch_size]
            try:
                ids = [self.vocab.encode(t, self.max_len) for t in chunk]
                batch = collate_ids(ids, pad_id=self.vocab.pad_id, min_len=self.min_len)
                with torch.inference_mode():
                    probs = torch.softmax(self.model(batch), dim=1)
            except RuntimeError as e:
                raise InferenceError("Forward pass failed", context={"batch": len(chunk)}) from e
            for row in probs.tolist():
                scores = dict(zip(self.labels, row, strict=True))
                best = max(scores, key=scores.__getitem__)
                out.append(Prediction(best, scores[best], scores))
        return out
