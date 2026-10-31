"""Loss function helpers for Transformer, Vision, and Segmentation models.

Provides cross-entropy, binary cross-entropy, segmentation loss, and KL divergence distillation losses.
"""

from __future__ import annotations

from typing import Any, Callable

import torch
import torch.nn.functional as F

Tensor = torch.Tensor


def language_modeling_cross_entropy(
    logits: Tensor,
    labels: Tensor,
    ignore_index: int = -100,
    label_smoothing: float = 0.0,
) -> Tensor:
    """Compute autoregressive token cross-entropy loss for language models.

    Args:
        logits (Tensor): Predicted token logits of shape `(B, T, V)`.
        labels (Tensor): Ground truth target token IDs of shape `(B, T)`.
        ignore_index (int, default=-100): Target token ID to ignore in loss calculation.
        label_smoothing (float, default=0.0): Label smoothing epsilon in `[0, 1)`.

    Returns:
        Tensor: Scalar cross-entropy loss.
    """
    if logits.ndim != 3:
        raise ValueError("Expected logits shape [B, T, V] for language modeling.")
    if labels.ndim != 2:
        raise ValueError("Expected labels shape [B, T] for language modeling.")

    return F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        labels.reshape(-1),
        ignore_index=ignore_index,
        label_smoothing=label_smoothing,
    )


def classification_cross_entropy(
    logits: Tensor,
    labels: Tensor,
    label_smoothing: float = 0.0,
) -> Tensor:
    """Compute multi-class cross-entropy loss for classification heads.

    Args:
        logits (Tensor): Class logits of shape `(B, C)`.
        labels (Tensor): Target class indices of shape `(B,)`.
        label_smoothing (float, default=0.0): Label smoothing coefficient.

    Returns:
        Tensor: Scalar classification loss.
    """
    return F.cross_entropy(logits, labels, label_smoothing=label_smoothing)


def binary_classification_bce_with_logits(logits: Tensor, labels: Tensor) -> Tensor:
    """Compute binary cross-entropy loss with logits for binary or multi-label classification.

    Args:
        logits (Tensor): Unnormalized logit predictions tensor.
        labels (Tensor): Target binary values tensor.

    Returns:
        Tensor: Scalar BCE loss.
    """
    return F.binary_cross_entropy_with_logits(logits, labels.float())


def segmentation_cross_entropy(
    logits: Tensor,
    labels: Tensor,
    ignore_index: int = 255,
) -> Tensor:
    """Compute 2D spatial cross-entropy loss for image segmentation tasks.

    Args:
        logits (Tensor): Predicted class logits of shape `(B, C, H, W)`.
        labels (Tensor): Target class mask of shape `(B, H, W)`.
        ignore_index (int, default=255): Masked label index to ignore.

    Returns:
        Tensor: Scalar segmentation loss.
    """
    return F.cross_entropy(logits, labels.long(), ignore_index=ignore_index)


def kl_divergence_distillation(
    student_logits: Tensor,
    teacher_logits: Tensor,
    temperature: float = 1.0,
) -> Tensor:
    """Compute Kullback-Leibler (KL) divergence distillation loss between student and teacher models.

    Args:
        student_logits (Tensor): Logit predictions from student model.
        teacher_logits (Tensor): Logit predictions from teacher model.
        temperature (float, default=1.0): Softmax temperature scaling factor (> 0).

    Returns:
        Tensor: Scalar KL divergence distillation loss.
    """
    if temperature <= 0:
        raise ValueError("temperature must be > 0")

    s = F.log_softmax(student_logits / temperature, dim=-1)
    t = F.softmax(teacher_logits / temperature, dim=-1)
    return F.kl_div(s, t, reduction="batchmean") * (temperature**2)


def get_loss_fn(name: str, **kwargs: Any) -> Callable[[Tensor, Tensor], Tensor]:
    """Factory for selecting loss function callables by string name.

    Args:
        name (str): Loss function identifier name.
        **kwargs (Any): Additional keyword arguments passed to loss function.

    Returns:
        Callable[[Tensor, Tensor], Tensor]: Loss calculation callable.
    """
    key = name.lower()

    if key == "lm_cross_entropy":
        return lambda logits, labels: language_modeling_cross_entropy(logits, labels, **kwargs)
    if key == "classification_cross_entropy":
        return lambda logits, labels: classification_cross_entropy(logits, labels, **kwargs)
    if key == "bce_with_logits":
        return lambda logits, labels: binary_classification_bce_with_logits(logits, labels)
    if key == "segmentation_cross_entropy":
        return lambda logits, labels: segmentation_cross_entropy(logits, labels, **kwargs)

    raise ValueError(f"Unsupported loss function: {name}")

