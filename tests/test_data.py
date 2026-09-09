"""Data pipeline: tokenizer edge cases, vocab round-trip, padding, and every DataError path."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
import torch
from conftest import LABELS, make_rows, write_dataset

from textclf.config import Config
from textclf.data import (
    DataBundle,
    TextDataset,
    Vocab,
    build_dataloaders,
    collate,
    collate_ids,
    load_labels,
    load_split,
    prepare_data,
    tokenize,
)
from textclf.exceptions import CheckpointError, DataError, ModelError

# ---------------------------------------------------------------- tokenizer


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("", []),
        ("   \n\t ", []),
        ("Hello, World!", ["hello", "world"]),
        ("It's 2024: GPT-4 beats 3.5", ["it's", "2024", "gpt", "4", "beats", "3", "5"]),
        ("Ünïcödé café — naïve", ["n", "c", "d", "caf", "na", "ve"]),  # ASCII-only by design
        ("!!! ... ???", []),
    ],
)
def test_tokenize(text: str, expected: list[str]) -> None:
    assert tokenize(text) == expected


# ---------------------------------------------------------------- vocab


def test_vocab_build_respects_min_freq_and_cap() -> None:
    texts = ["a a a b b c", "a b d"]
    v = Vocab.build(texts, min_freq=2, max_size=3)  # room for pad, unk + 1 token
    assert v.itos == ["<pad>", "<unk>", "a"]
    assert len(v) == 3
    v2 = Vocab.build(texts, min_freq=2, max_size=100)
    assert v2.itos == ["<pad>", "<unk>", "a", "b"]  # c and d are below min_freq


def test_vocab_encode_unknown_and_truncation() -> None:
    v = Vocab.build(["alpha beta gamma"], min_freq=1)
    ids = v.encode("alpha zzz gamma beta alpha", max_len=3)
    assert ids == [v.stoi["alpha"], v.unk_id, v.stoi["gamma"]]
    assert v.encode("", max_len=10) == []


def test_vocab_round_trip(tmp_path: Path) -> None:
    v = Vocab.build(["one two two three three three"], min_freq=1)
    v.save(tmp_path / "vocab.json")
    loaded = Vocab.load(tmp_path / "vocab.json")
    assert loaded.itos == v.itos
    assert loaded.stoi == v.stoi
    assert loaded.pad_id == 0 and loaded.unk_id == 1


def test_vocab_load_errors(tmp_path: Path) -> None:
    with pytest.raises(ModelError, match="Cannot load vocab") as info:
        Vocab.load(tmp_path / "missing.json")
    assert isinstance(info.value.__cause__, FileNotFoundError)

    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    with pytest.raises(ModelError, match="Cannot load vocab"):
        Vocab.load(bad)

    bad.write_text(json.dumps({"tokens": ["<pad>", "<unk>"]}))  # wrong key
    with pytest.raises(ModelError, match="Cannot load vocab"):
        Vocab.load(bad)

    bad.write_text(json.dumps({"itos": ["<pad>", 1]}))  # non-string token
    with pytest.raises(ModelError, match="unexpected layout"):
        Vocab.load(bad)

    bad.write_text(json.dumps({"itos": ["x", "y"]}))  # missing specials
    with pytest.raises(ModelError, match="must start with"):
        Vocab.load(bad)


def test_vocab_save_error_is_checkpoint_error(tmp_path: Path) -> None:
    v = Vocab.build(["a"], min_freq=1)
    with pytest.raises(CheckpointError) as info:
        v.save(tmp_path / "no_such_dir" / "vocab.json")
    assert info.value.context["path"].endswith("vocab.json")
    assert isinstance(info.value.__cause__, OSError)


# ---------------------------------------------------------------- collate


def test_collate_ids_pads_to_longest() -> None:
    batch = collate_ids([[5, 6, 7], [8]], pad_id=0, min_len=2)
    assert batch.shape == (2, 3)
    assert batch.tolist() == [[5, 6, 7], [8, 0, 0]]
    assert batch.dtype == torch.long


def test_collate_ids_respects_min_len_and_empty_sequences() -> None:
    batch = collate_ids([[1], []], pad_id=0, min_len=5)
    assert batch.shape == (2, 5)
    assert batch.tolist() == [[1, 0, 0, 0, 0], [0, 0, 0, 0, 0]]


def test_collate_returns_inputs_and_targets() -> None:
    inputs, targets = collate([([1, 2], 3), ([4], 0)], pad_id=0, min_len=1)
    assert inputs.tolist() == [[1, 2], [4, 0]]
    assert targets.tolist() == [3, 0]


def test_text_dataset_encodes_lazily() -> None:
    v = Vocab.build(["cat dog"], min_freq=1)
    ds = TextDataset(["cat dog fish", "dog"], [1, 0], v, max_len=2)
    assert len(ds) == 2
    assert ds[0] == ([v.stoi["cat"], v.stoi["dog"]], 1)
    with pytest.raises(DataError, match="length mismatch"):
        TextDataset(["a"], [0, 1], v, max_len=8)


# ---------------------------------------------------------------- load_split / load_labels


def test_load_split_happy_path(tiny_data_dir: Path) -> None:
    df = load_split(tiny_data_dir, "train", num_classes=4)
    assert list(df.columns) == ["text", "label"]
    assert len(df) == 50


def test_load_labels_errors(tmp_path: Path) -> None:
    with pytest.raises(DataError, match="Cannot read labels.json"):
        load_labels(tmp_path)
    (tmp_path / "labels.json").write_text(json.dumps(["only-one"]))
    with pytest.raises(DataError, match=">= 2 class names"):
        load_labels(tmp_path)


def test_load_split_missing_file(tmp_path: Path) -> None:
    with pytest.raises(DataError, match="Split file not found") as info:
        load_split(tmp_path, "train", num_classes=4)
    assert info.value.context["split"] == "train"


def test_load_split_corrupt_file(tmp_path: Path) -> None:
    (tmp_path / "train.parquet").write_bytes(b"definitely not parquet")
    with pytest.raises(DataError, match="Cannot read parquet"):
        load_split(tmp_path, "train", num_classes=4)


def _write(tmp_path: Path, df: pd.DataFrame) -> Path:
    df.to_parquet(tmp_path / "train.parquet", index=False)
    return tmp_path


def test_load_split_wrong_columns(tmp_path: Path) -> None:
    _write(tmp_path, pd.DataFrame({"content": ["a"], "label": [0]}))
    with pytest.raises(DataError, match="Unexpected columns") as info:
        load_split(tmp_path, "train", num_classes=4)
    assert info.value.context["missing"] == ["text"]


def test_load_split_empty(tmp_path: Path) -> None:
    _write(
        tmp_path,
        pd.DataFrame({"text": pd.Series([], dtype=str), "label": pd.Series([], dtype="int64")}),
    )
    with pytest.raises(DataError, match="Split is empty"):
        load_split(tmp_path, "train", num_classes=4)


def test_load_split_non_integer_labels(tmp_path: Path) -> None:
    _write(tmp_path, pd.DataFrame({"text": ["a", "b"], "label": ["0", "1"]}))
    with pytest.raises(DataError, match="must be integer"):
        load_split(tmp_path, "train", num_classes=4)


def test_load_split_empty_text(tmp_path: Path) -> None:
    _write(tmp_path, pd.DataFrame({"text": ["ok", "   "], "label": [0, 1]}))
    with pytest.raises(DataError, match="Empty texts") as info:
        load_split(tmp_path, "train", num_classes=4)
    assert info.value.context["empty_rows"] == 1


def test_load_split_label_out_of_range(tmp_path: Path) -> None:
    _write(tmp_path, pd.DataFrame({"text": ["a", "b", "c"], "label": [0, 4, -1]}))
    with pytest.raises(DataError, match="Labels out of range") as info:
        load_split(tmp_path, "train", num_classes=4)
    assert sorted(info.value.context["examples"]) == [-1, 4]


# ---------------------------------------------------------------- build_dataloaders


def _cfg(**train: object) -> Config:
    return Config.model_validate(
        {
            "data": {"num_workers": 0, "vocab_size": 1000, "min_freq": 1},
            "train": {"batch_size": 8, **train},
        }
    )


def test_build_dataloaders_shapes(tiny_data_dir: Path) -> None:
    bundle = build_dataloaders(_cfg(), tiny_data_dir)
    assert isinstance(bundle, DataBundle)
    assert bundle.labels == LABELS
    assert len(bundle.train_loader.dataset) == 45  # type: ignore[arg-type]  # Sized in practice
    assert len(bundle.val_loader.dataset) == 5  # type: ignore[arg-type]
    inputs, targets = next(iter(bundle.train_loader))
    assert inputs.shape[0] == 8 and inputs.shape[1] >= 5  # min_len = max kernel size (5)
    assert targets.shape == (8,)
    assert inputs.dtype == torch.long


def test_build_dataloaders_limit_and_seed(tiny_data_dir: Path) -> None:
    a = build_dataloaders(_cfg(), tiny_data_dir, limit=40)  # val = 4 rows = one per class
    b = build_dataloaders(_cfg(), tiny_data_dir, limit=40)
    assert len(a.train_loader.dataset) == 36  # type: ignore[arg-type]
    assert a.vocab.itos == b.vocab.itos  # same seed -> same sample -> same vocab
    with pytest.raises(DataError, match="limit must be"):
        build_dataloaders(_cfg(), tiny_data_dir, limit=0)


def test_build_dataloaders_too_few_rows_for_stratification(tmp_path: Path) -> None:
    make_rows(4, seed=3).to_parquet(tmp_path / "train.parquet", index=False)
    (tmp_path / "labels.json").write_text(json.dumps(LABELS))
    with pytest.raises(DataError, match="stratified validation split"):
        build_dataloaders(_cfg(), tmp_path)


# ---------------------------------------------------------------- prepare_data


def _fake_dataset_dict() -> object:
    from datasets import ClassLabel, Dataset, DatasetDict, Features, Value

    features = Features({"text": Value("string"), "label": ClassLabel(names=LABELS)})
    tr, te = make_rows(12), make_rows(8, seed=9)
    return DatasetDict(
        {
            "train": Dataset.from_dict(tr.to_dict(orient="list"), features=features),
            "test": Dataset.from_dict(te.to_dict(orient="list"), features=features),
        }
    )


def test_prepare_data_writes_artifacts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("textclf.data._download", lambda _id: _fake_dataset_dict())
    out = tmp_path / "ag"
    counts = prepare_data(out, "fake/ag_news")
    assert counts == {"train": 12, "test": 8}
    assert json.loads((out / "labels.json").read_text()) == LABELS
    df = load_split(out, "test", num_classes=4)
    assert str(df["label"].dtype) == "int64"
    assert len(df) == 8


def test_prepare_data_download_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def _boom(_id: str) -> object:
        raise ConnectionError("hub unreachable")

    monkeypatch.setattr("textclf.data._download", _boom)
    with pytest.raises(DataError) as info:
        prepare_data(tmp_path, "fake/ag_news")
    assert info.value.context == {
        "dataset_id": "fake/ag_news",
        "stage": "download",
        "out": str(tmp_path),
    }
    assert isinstance(info.value.__cause__, ConnectionError)


def test_prepare_data_missing_split(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from datasets import DatasetDict

    full = _fake_dataset_dict()
    assert isinstance(full, DatasetDict)
    monkeypatch.setattr("textclf.data._download", lambda _id: DatasetDict({"train": full["train"]}))
    with pytest.raises(DataError, match="lacks expected splits") as info:
        prepare_data(tmp_path, "fake/ag_news")
    assert info.value.context["missing"] == ["test"]


def test_write_dataset_helper_matches_loader(tmp_path: Path) -> None:
    """The fixture itself must satisfy the production validator."""
    write_dataset(tmp_path, n_train=8, n_test=4)
    assert len(load_split(tmp_path, "test", num_classes=4)) == 4
