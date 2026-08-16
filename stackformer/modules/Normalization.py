"""Normalization layers for StackFormer blocks with FP32 stability."""

from __future__ import annotations
import torch
import torch.nn as nn

class LayerNorm(nn.Module):
    def __init__(self, embed_dim: int, device: torch.device | str | None = None, dtype: torch.dtype | None = None, eps: float = 1e-5) -> None:
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
    def __init__(self, embed_dim: int, device: torch.device | str | None = None, dtype: torch.dtype | None = None, eps: float = 1e-5) -> None:
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