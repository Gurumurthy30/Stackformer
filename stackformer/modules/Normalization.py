"""Normalization layers for StackFormer blocks.

Provides per-token normalization operators used in Transformer architectures:
- LayerNorm (mean + variance normalization with affine scale/bias)
- RMSNorm (Root Mean Square normalization with affine scale only)

Equations:
- LayerNorm: y = gamma * (x - mean(x)) / sqrt(var(x) + eps) + beta
- RMSNorm:   y = gamma * x / (sqrt(mean(x^2)) + eps)
"""

from __future__ import annotations

import torch
import torch.nn as nn


class LayerNorm(nn.Module):
    """Layer normalization over the last tensor dimension.

    Computation:
        y = gamma * (x - mean(x)) / sqrt(var(x) + eps) + beta

    Constructor args:
        embed_dim (int): Normalized feature size ``C``.
        eps (float, default=1e-5): Numerical stability constant.
        device (torch.device | str | None, default=None): Target compute device.
        dtype (torch.dtype | None, default=None): Target data type.

    Learnable parameters:
        - weight (gamma): Scale parameter of shape ``(C,)``.
        - bias (beta): Shift parameter of shape ``(C,)``.

    Forward args:
        x (torch.Tensor): Input tensor of shape ``(B, T, C)``.

    Returns:
        torch.Tensor: Normalized output tensor of shape ``(B, T, C)``.

    Example:
        >>> norm = LayerNorm(embed_dim=256, eps=1e-5)
        >>> x = torch.randn(4, 32, 256)
        >>> y = norm(x)
        >>> y.shape
        torch.Size([4, 32, 256])
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
        self.weight = nn.Parameter(torch.ones(embed_dim, **factory_kwargs))  # gamma
        self.bias = nn.Parameter(torch.zeros(embed_dim, **factory_kwargs))  # beta

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C)
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        normalized_x = (x - mean) / torch.sqrt(var + self.eps)
        output = self.weight * normalized_x + self.bias
        return output  # (B, T, C)


class RMSNorm(nn.Module):
    """Root Mean Square Normalization (RMSNorm) over the last tensor dimension.

    Computation:
        y = gamma * x / (sqrt(mean(x^2)) + eps)

    Constructor args:
        embed_dim (int): Feature dimension size ``C``.
        eps (float, default=1e-5): Numerical stability constant.
        device (torch.device | str | None, default=None): Target compute device.
        dtype (torch.dtype | None, default=None): Target data type.

    Learnable parameters:
        - weight (gamma): Scale parameter of shape ``(C,)``.

    Forward args:
        x (torch.Tensor): Input tensor of shape ``(B, T, C)``.

    Returns:
        torch.Tensor: Normalized output tensor of shape ``(B, T, C)``.

    Example:
        >>> norm = RMSNorm(embed_dim=256)
        >>> x = torch.randn(4, 32, 256)
        >>> y = norm(x)
        >>> y.shape
        torch.Size([4, 32, 256])
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
        self.weight = nn.Parameter(torch.ones(embed_dim, **factory_kwargs))  # gamma

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C)
        rms = (x.pow(2).mean(-1, keepdim=True) + self.eps).sqrt()
        normalized_x = x / rms
        output = self.weight * normalized_x
        return output  # (B, T, C)


# Backward-compat aliases
LayerNormalization = LayerNorm
RMSNormalization = RMSNorm