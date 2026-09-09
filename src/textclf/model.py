"""TextCNN (Kim, 2014): embed -> parallel Conv1d -> max-over-time pooling -> dropout -> linear."""

from __future__ import annotations

import pickle

import torch
from torch import nn

from textclf.config import Config
from textclf.exceptions import ModelError

# Everything torch.load(weights_only=True) + load_state_dict can raise on a missing, truncated,
# foreign or mismatched file. Callers wrap these into CheckpointError / ModelError.
TORCH_LOAD_ERRORS = (OSError, KeyError, RuntimeError, ValueError, TypeError, pickle.UnpicklingError)


class TextCNN(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_classes: int,
        embed_dim: int = 128,
        num_filters: int = 100,
        kernel_sizes: tuple[int, ...] = (3, 4, 5),
        dropout: float = 0.3,
        pad_idx: int = 0,
    ) -> None:
        super().__init__()
        if not kernel_sizes:
            raise ModelError("At least one kernel size is required")
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.convs = nn.ModuleList(nn.Conv1d(embed_dim, num_filters, k) for k in kernel_sizes)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * len(kernel_sizes), num_classes)
        self.min_len = max(kernel_sizes)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:  # tokens: (B, L) int64
        if tokens.size(1) < self.min_len:
            raise ModelError(
                "Sequence shorter than the largest kernel",
                context={"len": tokens.size(1), "min_len": self.min_len},
            )
        x = self.embedding(tokens).transpose(1, 2)  # (B, E, L)
        pooled = [torch.relu(conv(x)).amax(dim=2) for conv in self.convs]  # each (B, F)
        logits: torch.Tensor = self.fc(self.dropout(torch.cat(pooled, dim=1)))  # (B, C)
        return logits


_REGISTRY: dict[str, type[nn.Module]] = {"textcnn": TextCNN}


def build_model(cfg: Config, vocab_size: int, num_classes: int) -> nn.Module:
    m = cfg.model
    try:
        factory = _REGISTRY[m.name]
    except KeyError as e:  # unreachable through Config (Literal), reachable through direct calls
        raise ModelError("Unknown model", context={"name": m.name, "known": list(_REGISTRY)}) from e
    return factory(
        vocab_size, num_classes, m.embed_dim, m.num_filters, tuple(m.kernel_sizes), m.dropout
    )


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
