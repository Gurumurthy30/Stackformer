"""Models package for StackFormer library architectures.

Exposes:
    - BaseModel: Common base architecture and lifecycle manager
    - GPT1, GPT2 (GPT_1, GPT_2): OpenAI GPT model constructors
    - Gemma1_2B, Gemma1_7B (gemma_1_2b, gemma_1_7b, Gemma_1_2B, Gemma_1_7B): Google Gemma architecture constructors
    - Llama1, Llama2 (llama_1, llama_2, Llama_1, Llama_2): Meta LLaMA architecture constructors
    - BERT, RoBERTa: Bidirectional encoder language models
    - Transformer: Standard Encoder-Decoder Transformer model architecture
    - ViT (vit): Vision Transformer architecture
    - SegFormer (segformer): Efficient transformer for semantic segmentation
    - list_models: Query available models grouped by category
    - get_model: Factory helper to instantiate models by name
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .base import BaseModel
from .language import (
    BERT,
    GPT1,
    GPT2,
    GPT_1,
    GPT_2,
    Gemma1_2B,
    Gemma1_7B,
    Gemma_1_2B,
    Gemma_1_7B,
    Llama1,
    Llama2,
    Llama_1,
    Llama_2,
    RoBERTa,
    Transformer,
    gemma_1_2b,
    gemma_1_7b,
    llama_1,
    llama_2,
)
from .vision import SegFormer, ViT, segformer, vit

if TYPE_CHECKING:
    import torch.nn as nn


def _canonicalize_model_name(name: str) -> str:
    """Normalize model architecture name to lowercase alphanumeric string."""
    return name.strip().lower().replace("-", "").replace("_", "").replace(" ", "")


_RAW_REGISTRY: dict[str, type[nn.Module]] = {
    # Language
    "BERT": BERT,
    "Gemma1_2B": Gemma1_2B,
    "Gemma1_7B": Gemma1_7B,
    "GPT1": GPT1,
    "GPT2": GPT2,
    "Llama1": Llama1,
    "Llama2": Llama2,
    "RoBERTa": RoBERTa,
    "Transformer": Transformer,
    # Vision
    "SegFormer": SegFormer,
    "ViT": ViT,
}

_MODEL_REGISTRY: dict[str, type[nn.Module]] = {}
for _name, _cls in _RAW_REGISTRY.items():
    _canon = _canonicalize_model_name(_name)
    if _canon in _MODEL_REGISTRY and _MODEL_REGISTRY[_canon] is not _cls:
        raise ValueError(
            f"Model registry collision: '{_name}' collides with '{_MODEL_REGISTRY[_canon]}'"
        )
    _MODEL_REGISTRY[_canon] = _cls


def list_models() -> dict[str, list[str]]:
    """Return dictionary of available models grouped by category.

    Returns:
        dict[str, list[str]]: Mapping from category ('language', 'vision')
        to sorted list of model names.
    """
    return {
        "language": [
            "BERT",
            "GPT1",
            "GPT2",
            "Gemma1_2B",
            "Gemma1_7B",
            "Llama1",
            "Llama2",
            "RoBERTa",
            "Transformer",
        ],
        "vision": [
            "SegFormer",
            "ViT",
        ],
    }


def get_model(name: str, **kwargs: Any) -> nn.Module:
    """Instantiate a model by name with the given configuration arguments.

    Args:
        name: Name of the model architecture (case-insensitive, e.g. 'gpt-2', 'gpt_2', 'GPT2', 'gpt2', ' GPT-2 ').
        **kwargs: Configuration arguments passed directly to the model constructor.

    Returns:
        nn.Module: Instantiated model instance.

    Raises:
        ValueError: If model name is not recognized.
    """
    canonical = _canonicalize_model_name(name)
    if canonical not in _MODEL_REGISTRY:
        available = sorted([m for cat in list_models().values() for m in cat])
        raise ValueError(
            f"Unknown model architecture '{name}'. Available models: {', '.join(available)}"
        )
    model_cls = _MODEL_REGISTRY[canonical]
    return model_cls(**kwargs)


__all__ = [
    "BERT",
    "BaseModel",
    "GPT1",
    "GPT2",
    "GPT_1",
    "GPT_2",
    "Gemma1_2B",
    "Gemma1_7B",
    "Gemma_1_2B",
    "Gemma_1_7B",
    "Llama1",
    "Llama2",
    "Llama_1",
    "Llama_2",
    "RoBERTa",
    "SegFormer",
    "Transformer",
    "ViT",
    "gemma_1_2b",
    "gemma_1_7b",
    "get_model",
    "list_models",
    "llama_1",
    "llama_2",
    "segformer",
    "vit",
]
