"""Data pipeline: download, validated parquet splits, tokenizer, vocabulary, batching.

The pure parts (tokenize, Vocab, collate_ids) have no pandas/datasets/sklearn dependency so the
serving image can import this module with only the base extras; the heavy libraries are imported
inside the functions that need them (train extras).
"""

from __future__ import annotations

import json
import logging
import re
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from torch.utils.data import DataLoader, Dataset

from textclf.config import Config
from textclf.exceptions import CheckpointError, DataError, ModelError
from textclf.retry import transient_retry

if TYPE_CHECKING:
    import pandas as pd
    from datasets import DatasetDict

log = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"[a-z0-9']+")
REQUIRED_COLUMNS = ("text", "label")
SPLITS = ("train", "test")


# --------------------------------------------------------------------------------------------
# Download / prepare
# --------------------------------------------------------------------------------------------
@transient_retry()
def _download(dataset_id: str) -> DatasetDict:
    """Fetch the dataset from the Hugging Face hub. Retried on network/throttling errors only."""
    from datasets import load_dataset

    ds = load_dataset(dataset_id)
    if not isinstance(ds, dict):  # a DatasetDict is a dict subclass; a bare Dataset is not
        raise TypeError(f"Expected a DatasetDict with splits, got {type(ds).__name__}")
    return ds


def prepare_data(out: Path, dataset_id: str = "fancyzhx/ag_news") -> dict[str, int]:
    """Write `train.parquet`, `test.parquet` and `labels.json` under `out`.

    Returns the row count per split. Every failure is a DataError carrying the stage it happened in.
    """
    stage = "download"
    try:
        ds = _download(dataset_id)
        stage = "convert"
        missing = [s for s in SPLITS if s not in ds]
        if missing:
            raise DataError("Dataset lacks expected splits", context={"missing": missing})
        labels: list[str] = list(ds["train"].features["label"].names)
        frames = {split: ds[split].to_pandas()[list(REQUIRED_COLUMNS)] for split in SPLITS}
        stage = "write"
        out.mkdir(parents=True, exist_ok=True)
        for split, frame in frames.items():
            frame["label"] = frame["label"].astype("int64")
            frame.to_parquet(out / f"{split}.parquet", index=False)
        (out / "labels.json").write_text(json.dumps(labels, indent=2))
    except (OSError, ValueError, TypeError, KeyError, AttributeError) as e:
        # OSError covers requests/HF hub errors (RequestException is an IOError) and disk failures;
        # ValueError/TypeError/KeyError/AttributeError cover an unexpected dataset schema.
        raise DataError(
            f"prepare-data failed during {stage}",
            context={"dataset_id": dataset_id, "stage": stage, "out": str(out)},
        ) from e
    counts = {split: int(len(frame)) for split, frame in frames.items()}
    log.info("prepared %s -> %s rows=%s labels=%s", dataset_id, out, counts, labels)
    return counts


# --------------------------------------------------------------------------------------------
# Tokenizer and vocabulary
# --------------------------------------------------------------------------------------------
def tokenize(text: str) -> list[str]:
    """Lowercase word/number tokens; punctuation and non-ASCII letters are dropped."""
    return _TOKEN_RE.findall(text.lower())


class Vocab:
    """Token -> id mapping built from the TRAINING split only. 0 = <pad>, 1 = <unk>."""

    PAD, UNK = "<pad>", "<unk>"
    pad_id, unk_id = 0, 1

    def __init__(self, itos: list[str]) -> None:
        if itos[:2] != [self.PAD, self.UNK]:
            raise ModelError("Vocab must start with <pad>, <unk>", context={"head": itos[:2]})
        self.itos = itos
        self.stoi = {tok: i for i, tok in enumerate(itos)}

    def __len__(self) -> int:
        return len(self.itos)

    @classmethod
    def build(cls, texts: Iterable[str], min_freq: int = 2, max_size: int = 30_000) -> Vocab:
        counter: Counter[str] = Counter()
        for text in texts:
            counter.update(tokenize(text))
        kept = [t for t, c in counter.most_common() if c >= min_freq]
        return cls([cls.PAD, cls.UNK, *kept[: max_size - 2]])

    def encode(self, text: str, max_len: int) -> list[int]:
        return [self.stoi.get(t, self.unk_id) for t in tokenize(text)][:max_len]

    def save(self, path: Path) -> None:
        try:
            path.write_text(json.dumps({"itos": self.itos}))
        except OSError as e:
            raise CheckpointError("Cannot write vocab", context={"path": str(path)}) from e

    @classmethod
    def load(cls, path: Path) -> Vocab:
        try:
            payload = json.loads(path.read_text())
            itos = payload["itos"]
        except (OSError, ValueError, KeyError, TypeError) as e:
            raise ModelError("Cannot load vocab", context={"path": str(path)}) from e
        if not isinstance(itos, list) or not all(isinstance(t, str) for t in itos):
            raise ModelError("Vocab file has an unexpected layout", context={"path": str(path)})
        return cls(itos)


# --------------------------------------------------------------------------------------------
# Validated split loading
# --------------------------------------------------------------------------------------------
def load_labels(data_dir: Path) -> list[str]:
    path = data_dir / "labels.json"
    try:
        labels = json.loads(path.read_text())
    except (OSError, ValueError) as e:
        raise DataError("Cannot read labels.json", context={"path": str(path)}) from e
    if (
        not isinstance(labels, list)
        or len(labels) < 2
        or not all(isinstance(x, str) for x in labels)
    ):
        raise DataError(
            "labels.json must be a list of >= 2 class names", context={"path": str(path)}
        )
    return labels


def load_split(data_dir: Path, split: str, num_classes: int) -> pd.DataFrame:
    """Read `<split>.parquet` and fail early on anything the model could not train on."""
    import pandas as pd

    path = data_dir / f"{split}.parquet"
    ctx: dict[str, Any] = {"path": str(path), "split": split}
    if not path.is_file():
        raise DataError("Split file not found; run `textclf prepare-data`", context=ctx)
    try:
        df = pd.read_parquet(path)
    except (OSError, ValueError) as e:  # pyarrow.ArrowInvalid is a ValueError
        raise DataError("Cannot read parquet file", context=ctx) from e

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise DataError(
            "Unexpected columns", context={**ctx, "missing": missing, "found": list(df.columns)}
        )
    if df.empty:
        raise DataError("Split is empty", context=ctx)
    if not pd.api.types.is_integer_dtype(df["label"]):
        raise DataError(
            "label column must be integer", context={**ctx, "dtype": str(df["label"].dtype)}
        )
    if not df["text"].map(lambda v: isinstance(v, str)).all():
        raise DataError("text column must contain only strings", context=ctx)
    n_empty = int((df["text"].str.strip() == "").sum())
    if n_empty:
        raise DataError("Empty texts found", context={**ctx, "empty_rows": n_empty})
    bad = df.loc[~df["label"].between(0, num_classes - 1), "label"]
    if not bad.empty:
        raise DataError(
            "Labels out of range",
            context={**ctx, "num_classes": num_classes, "examples": bad.unique()[:5].tolist()},
        )
    return df


# --------------------------------------------------------------------------------------------
# Dataset and batching
# --------------------------------------------------------------------------------------------
class TextDataset(Dataset[tuple[list[int], int]]):
    """Encodes lazily so a `limit` slice never pays for the whole corpus."""

    def __init__(self, texts: list[str], labels: list[int], vocab: Vocab, max_len: int) -> None:
        if len(texts) != len(labels):
            raise DataError(
                "texts/labels length mismatch", context={"texts": len(texts), "labels": len(labels)}
            )
        self.texts, self.labels, self.vocab, self.max_len = texts, labels, vocab, max_len

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> tuple[list[int], int]:
        return self.vocab.encode(self.texts[idx], self.max_len), self.labels[idx]


def collate_ids(ids: list[list[int]], pad_id: int, min_len: int) -> torch.Tensor:
    """Right-pad to max(longest, min_len): Conv1d never sees a sequence shorter than its kernel."""
    width = max(max((len(seq) for seq in ids), default=0), min_len)
    batch = torch.full((len(ids), width), pad_id, dtype=torch.long)
    for row, seq in enumerate(ids):
        if seq:
            batch[row, : len(seq)] = torch.tensor(seq, dtype=torch.long)
    return batch


def collate(
    batch: list[tuple[list[int], int]], pad_id: int, min_len: int
) -> tuple[torch.Tensor, torch.Tensor]:
    ids = [seq for seq, _ in batch]
    targets = torch.tensor([label for _, label in batch], dtype=torch.long)
    return collate_ids(ids, pad_id, min_len), targets


@dataclass(frozen=True)
class DataBundle:
    train_loader: DataLoader[tuple[list[int], int]]
    val_loader: DataLoader[tuple[list[int], int]]
    vocab: Vocab
    labels: list[str]


def build_dataloaders(cfg: Config, data_dir: Path, limit: int | None = None) -> DataBundle:
    """Seeded, stratified train/val split from the training parquet; vocab from train rows only."""
    from sklearn.model_selection import train_test_split

    labels = load_labels(data_dir)
    df = load_split(data_dir, "train", num_classes=len(labels))
    if limit is not None:
        if limit < 1:
            raise DataError("limit must be >= 1", context={"limit": limit})
        df = df.sample(n=min(limit, len(df)), random_state=cfg.train.seed)
    texts, targets = df["text"].tolist(), df["label"].astype(int).tolist()
    try:
        tr_x, va_x, tr_y, va_y = train_test_split(
            texts,
            targets,
            test_size=cfg.data.val_fraction,
            stratify=targets,
            random_state=cfg.train.seed,
        )
    except ValueError as e:  # too few rows per class for a stratified split
        raise DataError(
            "Cannot build a stratified validation split",
            context={"rows": len(texts), "val_fraction": cfg.data.val_fraction},
        ) from e

    vocab = Vocab.build(tr_x, min_freq=cfg.data.min_freq, max_size=cfg.data.vocab_size)
    min_len = max(cfg.model.kernel_sizes)
    collate_fn = partial(collate, pad_id=vocab.pad_id, min_len=min_len)
    pin = torch.cuda.is_available()
    generator = torch.Generator().manual_seed(cfg.train.seed)
    train_loader = DataLoader(
        TextDataset(tr_x, tr_y, vocab, cfg.data.max_len),
        batch_size=cfg.train.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=cfg.data.num_workers,
        pin_memory=pin,
        generator=generator,
    )
    val_loader = DataLoader(
        TextDataset(va_x, va_y, vocab, cfg.data.max_len),
        batch_size=cfg.train.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=cfg.data.num_workers,
        pin_memory=pin,
    )
    log.info(
        "data: train=%d val=%d vocab=%d classes=%d limit=%s",
        len(tr_x),
        len(va_x),
        len(vocab),
        len(labels),
        limit,
    )
    return DataBundle(train_loader, val_loader, vocab, labels)
