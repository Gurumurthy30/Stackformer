"""Optimization package providing optimizer factories, LR schedulers, and loss functions.

Exposes:
    - create_optimizer: Factory for constructing AdamW, Adam, SGD, and Adafactor optimizers
    - create_scheduler: Factory for constructing cosine, linear, step, and constant schedulers
    - get_parameter_groups: Helper to separate decay and no-decay parameter groups
    - language_modeling_cross_entropy, classification_cross_entropy: Loss functions
    - binary_classification_bce_with_logits, segmentation_cross_entropy, kl_divergence_distillation: Loss functions
    - get_loss_fn: Loss function factory by name string
"""

from .factories import create_optimizer, create_scheduler, get_parameter_groups
from .loss_fn import (
    binary_classification_bce_with_logits,
    classification_cross_entropy,
    get_loss_fn,
    kl_divergence_distillation,
    language_modeling_cross_entropy,
    segmentation_cross_entropy,
)

__all__ = [
    "create_optimizer",
    "create_scheduler",
    "get_parameter_groups",
    "language_modeling_cross_entropy",
    "classification_cross_entropy",
    "binary_classification_bce_with_logits",
    "segmentation_cross_entropy",
    "kl_divergence_distillation",
    "get_loss_fn",
]

