"""Typed configuration. YAML for training hyperparameters, environment variables for serving."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict

from textclf.exceptions import ConfigError


class _Strict(BaseModel):
    # A typo in YAML is a ConfigError, not a silent default.
    model_config = ConfigDict(extra="forbid")


class DataConfig(_Strict):
    dataset_id: str = "fancyzhx/ag_news"  # legacy alias: "ag_news"
    max_len: int = Field(128, ge=8, le=1024)
    vocab_size: int = Field(30_000, ge=1_000)
    min_freq: int = Field(2, ge=1)
    val_fraction: float = Field(0.1, gt=0, lt=0.5)
    num_workers: int = Field(2, ge=0)


class ModelConfig(_Strict):
    name: Literal["textcnn"] = "textcnn"
    embed_dim: int = Field(128, ge=8)
    num_filters: int = Field(100, ge=1)
    kernel_sizes: list[int] = [3, 4, 5]
    dropout: float = Field(0.3, ge=0, lt=1)


class TrainConfig(_Strict):
    epochs: int = Field(5, ge=1)
    batch_size: int = Field(64, ge=1)
    lr: float = Field(2e-3, gt=0)
    weight_decay: float = Field(0.01, ge=0)
    max_grad_norm: float = Field(1.0, gt=0)
    amp: bool = True
    seed: int = 42
    early_stopping_patience: int = Field(2, ge=1)


class Config(_Strict):
    data: DataConfig = DataConfig()
    model: ModelConfig = ModelConfig()
    train: TrainConfig = TrainConfig()


def load_config(path: Path) -> Config:
    try:
        raw = yaml.safe_load(path.read_text())
    except FileNotFoundError as e:
        raise ConfigError("Config file not found", context={"path": str(path)}) from e
    except yaml.YAMLError as e:
        raise ConfigError("Malformed YAML", context={"path": str(path)}) from e
    try:
        return Config.model_validate(raw or {})
    except ValidationError as e:
        raise ConfigError(f"Invalid config values:\n{e}", context={"path": str(path)}) from e


class ServeSettings(BaseSettings):
    """Serving configuration from environment variables (prefix TEXTCLF_)."""

    model_config = SettingsConfigDict(env_prefix="TEXTCLF_", env_file=".env", extra="ignore")

    model_uri: str = "file://outputs/model"
    host: str = "127.0.0.1"  # the container sets 0.0.0.0 explicitly; secure default otherwise
    port: int = 8000
    workers: int = 1
    max_batch_size: int = Field(64, ge=1)
    torch_threads: int = Field(2, ge=1)
    log_level: str = "INFO"
