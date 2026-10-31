"""Normalization layers for StackFormer blocks with FP32 stability.

Provides `LayerNorm` and `RMSNorm` (Root Mean Square Normalization) modules with FP32 accumulation
for numerical stability during training and inference.

RMSNorm formula:
    ``y = gamma * x / sqrt(mean(x^2) + eps)``
"""

from __future__ import annotations

import torch
import torch.nn as nn


class LayerNorm(nn.Module):
    """Standard Layer Normalization module with FP32 stability.

    Simple explanation:
        Normalizes input features across the last dimension to zero mean and unit variance,
        then scales and shifts using learnable weight (gamma) and bias (beta) parameters.

    Constructor args:
        embed_dim (int): Dimensionality of the feature space to normalize (C).
        device (torch.device | str | None, default=None): Target compute device.
        dtype (torch.dtype | None, default=None): Target data type.
        eps (float, default=1e-5): Epsilon added to variance for numerical stability.

    Learnable parameters:
        weight: Shape ``(C,)``. Scale parameter (gamma).
        bias: Shape ``(C,)``. Shift parameter (beta).

    Forward args:
        x (torch.Tensor): Input tensor of shape ``(B, ..., C)``.

    Returns:
        torch.Tensor: Normalized tensor of shape ``(B, ..., C)``.

    Example:
        >>> norm = LayerNorm(768)
        >>> x = torch.randn(2, 10, 768)
        >>> y = norm(x)
        >>> y.shape
        torch.Size([2, 10, 768])
    """

    def __init__(
        self,
        embed_dim: int,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
        eps: float = 1e-5,
    ) -> None:
        super().__init__()
        self.eps = eps
        factory_kwargs = {"device": device, "dtype": dtype}
        self.weight = nn.Parameter(torch.ones(embed_dim, **factory_kwargs))
        self.bias = nn.Parameter(torch.zeros(embed_dim, **factory_kwargs))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        x32 = x.float()
        mean = x32.mean(dim=-1, keepdim=True)
        var = x32.var(dim=-1, keepdim=True, unbiased=False)
        normalized_x = (x32 - mean) / torch.sqrt(var + self.eps)
        output = self.weight.float() * normalized_x + self.bias.float()
        return output.to(orig_dtype)


class RMSNorm(nn.Module):
    """Root Mean Square Normalization (RMSNorm) module with FP32 stability.

    Simple explanation:
        Normalizes input features by their root-mean-square without mean centering,
        reducing computational overhead compared to standard LayerNorm.

    Constructor args:
        embed_dim (int): Dimensionality of the feature space to normalize (C).
        device (torch.device | str | None, default=None): Target compute device.
        dtype (torch.dtype | None, default=None): Target data type.
        eps (float, default=1e-5): Epsilon added inside square root for numerical stability.

    Learnable parameters:
        weight: Shape ``(C,)``. Scale parameter (gamma).

    Forward args:
        x (torch.Tensor): Input tensor of shape ``(B, ..., C)``.

    Returns:
        torch.Tensor: Normalized tensor of shape ``(B, ..., C)``.

    Example:
        >>> norm = RMSNorm(768)
        >>> x = torch.randn(2, 10, 768)
        >>> y = norm(x)
        >>> y.shape
        torch.Size([2, 10, 768])
    """

    def __init__(
        self,
        embed_dim: int,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
        eps: float = 1e-5,
    ) -> None:
        super().__init__()
        self.eps = eps
        factory_kwargs = {"device": device, "dtype": dtype}
        self.weight = nn.Parameter(torch.ones(embed_dim, **factory_kwargs))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        x32 = x.float()
        rms = (x32.pow(2).mean(-1, keepdim=True) + self.eps).sqrt()
        normalized_x = x32 / rms
        output = self.weight.float() * normalized_x
        return output.to(orig_dtype)


LayerNormalization = LayerNorm
RMSNormalization = RMSNorm