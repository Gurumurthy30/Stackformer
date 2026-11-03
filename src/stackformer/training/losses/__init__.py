"""Loss functions registry and built-in loss implementations."""

from __future__ import annotations

import functools
from typing import Any, Callable
import torch
import torch.nn.functional as F

_LOSS_REGISTRY: dict[str, Callable[[Any, Any], torch.Tensor]] = {}


def register_loss(name: str):
    """Decorator to register a custom loss function under a given name."""

    def decorator(fn: Callable[[Any, Any], torch.Tensor]):
        _LOSS_REGISTRY[name.lower().strip()] = fn
        return fn

    return decorator


def get_loss(
    loss_name_or_fn: str | Callable[[Any, Any], torch.Tensor],
    loss_mode: str = "auto",
) -> Callable[[Any, Any], torch.Tensor]:
    """Retrieve a loss function by name or return the callable directly.

    Args:
        loss_name_or_fn: Registered loss name or callable function.
        loss_mode: Strategy for resolving ambiguous targets ("auto", "lm", "classification").

    Returns:
        Callable[[Any, Any], torch.Tensor]: Configured loss function.
    """
    if callable(loss_name_or_fn):
        return loss_name_or_fn
    name = loss_name_or_fn.lower().strip()
    if name not in _LOSS_REGISTRY:
        available = sorted(_LOSS_REGISTRY.keys())
        raise ValueError(f"Unknown loss '{loss_name_or_fn}'. Available registered losses: {available}")
    fn = _LOSS_REGISTRY[name]
    if name == "cross_entropy" and loss_mode != "auto":
        return functools.partial(fn, loss_mode=loss_mode)
    return fn


def _extract_target(batch: Any) -> torch.Tensor:
    """Helper to extract target labels or values from a batch."""
    if isinstance(batch, (tuple, list)):
        if len(batch) >= 2:
            return batch[1]
        return batch[0]
    elif isinstance(batch, dict):
        for key in ["labels", "targets", "target", "label", "input_ids"]:
            if key in batch:
                return batch[key]
    elif isinstance(batch, torch.Tensor):
        return batch
    raise ValueError(f"Could not extract target tensor from batch of type {type(batch).__name__}")


@register_loss("cross_entropy")
def cross_entropy_loss(
    outputs: torch.Tensor,
    batch: Any,
    ignore_index: int = -100,
    loss_mode: str = "auto",
) -> torch.Tensor:
    """General cross entropy loss supporting next-token LLM shift and classification.

    Resolution logic:
    - loss_mode == "lm":
        Enforces autoregressive next-token prediction shift:
        logits[:, :-1, :] vs labels[:, 1:].
        Requires 3D logits (B, T, V) and 2D targets (B, T).
    - loss_mode == "classification":
        Enforces standard classification without shifting.
        For 2D logits (B, C) vs 1D targets (B,), standard cross entropy is applied.
        For 3D logits (B, N, C) vs 2D targets (B, N) (e.g. dense token classification,
        masked LM without shift, semantic segmentation), unshifted token-level cross entropy is applied.
    - loss_mode == "auto":
        Inspects output and target tensor dimensionality:
        * 3D logits (B, T, V) and 2D targets (B, T): Treated as causal language modeling;
          automatically shifts logits[:, :-1, :] against targets[:, 1:].
        * 2D logits (B, C) and 1D targets (B,): Treated as standard classification (no shift).
        * Ambiguous case: When targets is 2D and outputs is 3D, 'auto' selects next-token-shift.
          If non-shifted token classification is desired instead, pass loss_mode="classification".
    """
    targets = _extract_target(batch)

    if loss_mode == "lm":
        if outputs.ndim != 3 or targets.ndim != 2:
            raise ValueError(
                f"loss_mode='lm' requires 3D logits (B, T, V) and 2D targets (B, T), "
                f"got outputs.ndim={outputs.ndim}, targets.ndim={targets.ndim}"
            )
        shift_logits = outputs[:, :-1, :].contiguous()
        shift_labels = targets[:, 1:].contiguous()
        return F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=ignore_index,
        )

    if loss_mode == "classification":
        if outputs.ndim == 2:
            return F.cross_entropy(outputs, targets, ignore_index=ignore_index)
        if outputs.ndim == 4 and targets.ndim == 3:
            return F.cross_entropy(outputs, targets, ignore_index=ignore_index)
        return F.cross_entropy(
            outputs.view(-1, outputs.size(-1)),
            targets.view(-1),
            ignore_index=ignore_index,
        )

    # loss_mode == "auto"
    if outputs.ndim == 3 and targets.ndim == 2:
        # LLM causal next-token prediction
        shift_logits = outputs[:, :-1, :].contiguous()
        shift_labels = targets[:, 1:].contiguous()
        return F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=ignore_index,
        )
    elif outputs.ndim == 2:
        # Standard classification
        return F.cross_entropy(outputs, targets, ignore_index=ignore_index)
    elif outputs.ndim == 4 and targets.ndim == 3:
        # 2D Semantic segmentation
        return F.cross_entropy(outputs, targets, ignore_index=ignore_index)
    else:
        # Flattened or higher-dim fallback
        return F.cross_entropy(
            outputs.view(-1, outputs.size(-1)),
            targets.view(-1),
            ignore_index=ignore_index,
        )


@register_loss("mse")
def mse_loss(outputs: torch.Tensor, batch: Any) -> torch.Tensor:
    """Mean Squared Error loss."""
    targets = _extract_target(batch)
    return F.mse_loss(outputs, targets)


@register_loss("l1")
def l1_loss(outputs: torch.Tensor, batch: Any) -> torch.Tensor:
    """L1 / Mean Absolute Error loss."""
    targets = _extract_target(batch)
    return F.l1_loss(outputs, targets)


@register_loss("huber")
def huber_loss(outputs: torch.Tensor, batch: Any, delta: float = 1.0) -> torch.Tensor:
    """Huber / Smooth L1 loss."""
    targets = _extract_target(batch)
    return F.smooth_l1_loss(outputs, targets, beta=delta)


@register_loss("bce")
def bce_loss(outputs: torch.Tensor, batch: Any) -> torch.Tensor:
    """Binary Cross Entropy loss."""
    targets = _extract_target(batch)
    return F.binary_cross_entropy(outputs, targets.float())


@register_loss("bce_logits")
def bce_logits_loss(outputs: torch.Tensor, batch: Any) -> torch.Tensor:
    """Binary Cross Entropy with Logits loss."""
    targets = _extract_target(batch)
    return F.binary_cross_entropy_with_logits(outputs, targets.float())


@register_loss("dice")
def dice_loss(outputs: torch.Tensor, batch: Any, eps: float = 1e-6) -> torch.Tensor:
    """Dice loss for binary or multi-class segmentation."""
    targets = _extract_target(batch)
    probs = torch.softmax(outputs, dim=1) if outputs.size(1) > 1 else torch.sigmoid(outputs)
    if targets.ndim == outputs.ndim - 1:
        # Convert integer targets to one-hot
        num_classes = outputs.size(1)
        targets_one_hot = F.one_hot(targets.clamp(min=0), num_classes=num_classes)
        # Move channel dim to dim 1
        targets_one_hot = targets_one_hot.permute(0, -1, *range(1, targets.ndim)).float()
    else:
        targets_one_hot = targets.float()

    dims = tuple(range(2, outputs.ndim))
    intersection = torch.sum(probs * targets_one_hot, dim=dims)
    cardinality = torch.sum(probs + targets_one_hot, dim=dims)
    dice_score = (2.0 * intersection + eps) / (cardinality + eps)
    return 1.0 - torch.mean(dice_score)


@register_loss("focal")
def focal_loss(
    outputs: torch.Tensor, batch: Any, gamma: float = 2.0, alpha: float | None = 0.25, ignore_index: int = -100
) -> torch.Tensor:
    """Focal Loss for addressing class imbalance."""
    targets = _extract_target(batch)
    ce_loss = F.cross_entropy(outputs, targets, reduction="none", ignore_index=ignore_index)
    pt = torch.exp(-ce_loss)
    focal = ((1.0 - pt) ** gamma) * ce_loss
    if alpha is not None:
        focal = alpha * focal
    return focal.mean()


@register_loss("seg_cross_entropy")
def seg_cross_entropy_loss(outputs: torch.Tensor, batch: Any, ignore_index: int = -100) -> torch.Tensor:
    """Segmentation cross entropy with automatic 2D spatial upsampling if needed."""
    targets = _extract_target(batch)
    if outputs.ndim == 4 and targets.ndim == 3:
        # Spatial dimensions mismatch (e.g. SegFormer 1/4 resolution decode head)
        if outputs.shape[-2:] != targets.shape[-2:]:
            outputs = F.interpolate(outputs, size=targets.shape[-2:], mode="bilinear", align_corners=False)
    return F.cross_entropy(outputs, targets, ignore_index=ignore_index)


# =========================================================================
# Extension point and placeholder for custom kernel-backed cross entropy
# (e.g. fused, chunked, or Triton-backed cross entropy implementations)
# =========================================================================


def set_kernel_cross_entropy(kernel_fn: Callable[[Any, Any], torch.Tensor]) -> None:
    """Register or override the default cross entropy with a custom fused/Triton kernel."""
    register_loss("cross_entropy")(kernel_fn)


__all__ = [
    "register_loss",
    "get_loss",
    "cross_entropy_loss",
    "mse_loss",
    "l1_loss",
    "huber_loss",
    "bce_loss",
    "bce_logits_loss",
    "dice_loss",
    "focal_loss",
    "seg_cross_entropy_loss",
    "set_kernel_cross_entropy",
]
