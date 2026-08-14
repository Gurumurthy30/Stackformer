"""Normalization layers for StackFormer blocks.

Provides per-token normalization operators used in Transformer architectures:
- LayerNorm (mean + variance normalization with affine scale/bias)
- RMSNorm (Root Mean Square normalization with affine scale only)

Equations:
- LayerNorm: y = gamma * (x - mean(x)) / sqrt(var(x) + eps) + beta
- RMSNorm:   y = gamma * x / (sqrt(mean(x^2)) + eps)

Numerical stability:
- Both layers upcast the input to float32 for the mean/variance/sqrt math and cast
  the result back to the input's original dtype before returning. This mirrors how
  PyTorch's built-in `torch.nn.LayerNorm` is special-cased by `torch.autocast`
  (always executed in float32 internally). Without this upcast, plain tensor ops
  like `.var()` and `.sqrt()` are NOT protected by autocast, so under fp16 mixed
  precision they run in fp16. Deep pre-norm transformer stacks routinely produce
  activations with magnitude in the hundreds by their later layers, and squaring
  values like that inside `var()` already exceeds fp16's ~65504 max, silently
  overflowing to `inf`. That `inf` poisons the row's mean/variance, `x - mean`
  becomes `NaN`, and it propagates through every downstream layer -- surfacing as
  `NaN`/`Inf` training loss. Upcasting internally closes this failure mode while
  keeping the public dtype contract (and memory footprint) of each call unchanged.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class LayerNorm(nn.Module):
    """Layer normalization over the last tensor dimension.

    Computation:
        y = gamma * (x - mean(x)) / sqrt(var(x) + eps) + beta

    The mean/variance/normalization math is always performed in float32
    internally (regardless of the input's dtype) and cast back to the input's
    original dtype before returning, to avoid fp16 overflow in `var()`/`sqrt()`
    on large-magnitude activations. See module docstring for details.

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
        torch.Tensor: Normalized output tensor of shape ``(B, T, C)``, same dtype as input.

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
    """Root Mean Square Normalization (RMSNorm) over the last tensor dimension.

    Computation:
        y = gamma * x / (sqrt(mean(x^2)) + eps)

    The RMS/normalization math is always performed in float32 internally
    (regardless of the input's dtype) and cast back to the input's original
    dtype before returning, for the same fp16-overflow reasons as `LayerNorm`
    above -- squaring large activations in fp16 can overflow before the sqrt
    is ever taken.

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
        torch.Tensor: Normalized output tensor of shape ``(B, T, C)``, same dtype as input.

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
            self.weight = nn.Parameter(torch.ones(embed_dim, **factory_kwargs))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        x32 = x.float()
        rms = (x32.pow(2).mean(-1, keepdim=True) + self.eps).sqrt()
        normalized_x = x32 / rms
        output = self.weight.float() * normalized_x
        return output.to(orig_dtype)

# Backward-compat aliases
LayerNormalization = LayerNorm
RMSNormalization = RMSNorm