"""Stackformer data ingestion subsystem (placeholder / design stub).

Provides protocols and foundational interfaces for pre-training, SFT,
and multimodal data loading pipelines.
"""

from __future__ import annotations

from typing import Any, Iterator, Protocol, Sequence, runtime_checkable
import torch


@runtime_checkable
class Dataset(Protocol):
    """Protocol for map-style or iterable dataset primitives."""

    def __iter__(self) -> Iterator[dict[str, Any]]:
        ...


class DataCollator:
    """Base collator interface for converting sample lists into batched tensors."""

    def __call__(self, batch: Sequence[dict[str, Any]]) -> dict[str, torch.Tensor]:
        raise NotImplementedError("DataCollator implementations planned for Phase 4.")


class PackedSequenceCollator(DataCollator):
    """Collates variable-length documents into packed fixed-length sequences."""

    def __init__(self, max_seq_len: int, pad_token_id: int = 0) -> None:
        self.max_seq_len = max_seq_len
        self.pad_token_id = pad_token_id

    def __call__(self, batch: Sequence[dict[str, Any]]) -> dict[str, torch.Tensor]:
        raise NotImplementedError("Packed sequence collation planned for Phase 4.")


__all__ = [
    "DataCollator",
    "Dataset",
    "PackedSequenceCollator",
]
