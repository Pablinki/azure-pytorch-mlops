"""Shared fixtures: a tiny synthetic dataset and a tiny trained model (offline, fast)."""

from __future__ import annotations

import json
import random
from pathlib import Path

import mlflow
import pandas as pd
import pytest

from textclf.config import Config

LABELS = ["World", "Sports", "Business", "Sci/Tech"]
_WORDS = {
    0: ["election", "minister", "border", "treaty", "capital", "parliament"],
    1: ["goal", "match", "coach", "season", "league", "striker"],
    2: ["shares", "profit", "market", "investors", "quarter", "revenue"],
    3: ["software", "chip", "rocket", "research", "algorithm", "satellite"],
}

TINY_CONFIG = {
    "data": {"num_workers": 0, "vocab_size": 1000, "min_freq": 1, "max_len": 16},
    "model": {"embed_dim": 16, "num_filters": 8, "kernel_sizes": [2, 3]},
    "train": {"epochs": 2, "batch_size": 8, "lr": 0.02, "early_stopping_patience": 5, "seed": 7},
}


def make_rows(n: int, seed: int = 0) -> pd.DataFrame:
    rng = random.Random(seed)  # noqa: S311  # synthetic test data, not security
    rows = []
    for i in range(n):
        label = i % len(LABELS)
        words = rng.choices(_WORDS[label], k=rng.randint(4, 9))
        rows.append({"text": " ".join(words).capitalize() + ".", "label": label})
    return pd.DataFrame(rows)


def write_dataset(path: Path, n_train: int = 50, n_test: int = 20) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    make_rows(n_train, seed=1).to_parquet(path / "train.parquet", index=False)
    make_rows(n_test, seed=2).to_parquet(path / "test.parquet", index=False)
    (path / "labels.json").write_text(json.dumps(LABELS))
    return path


def tiny_config() -> Config:
    return Config.model_validate(TINY_CONFIG)


@pytest.fixture(scope="session")
def tiny_data_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """50 synthetic training rows, 20 test rows, 4 balanced classes."""
    return write_dataset(tmp_path_factory.mktemp("data"))


@pytest.fixture(scope="session", autouse=True)
def mlflow_tmp_tracking(tmp_path_factory: pytest.TempPathFactory) -> None:
    """Keep MLflow runs created by tests out of the repo's mlruns.db."""
    db = tmp_path_factory.mktemp("mlruns") / "mlflow.db"
    mlflow.set_tracking_uri(f"sqlite:///{db.as_posix()}")


@pytest.fixture(scope="session")
def tiny_model_dir(tiny_data_dir: Path, tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Train the TextCNN for 2 epochs on the synthetic data and export the five artifact files."""
    from textclf.train import run_training

    out = tmp_path_factory.mktemp("run")
    run_training(tiny_config(), tiny_data_dir, out)
    return out / "model"
