"""Shared fixtures: a tiny synthetic dataset (offline, fast) that the model can actually learn."""

from __future__ import annotations

import json
import random
from pathlib import Path

import pandas as pd
import pytest

LABELS = ["World", "Sports", "Business", "Sci/Tech"]
_WORDS = {
    0: ["election", "minister", "border", "treaty", "capital", "parliament"],
    1: ["goal", "match", "coach", "season", "league", "striker"],
    2: ["shares", "profit", "market", "investors", "quarter", "revenue"],
    3: ["software", "chip", "rocket", "research", "algorithm", "satellite"],
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


@pytest.fixture(scope="session")
def tiny_data_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """50 synthetic training rows, 20 test rows, 4 balanced classes."""
    return write_dataset(tmp_path_factory.mktemp("data"))
