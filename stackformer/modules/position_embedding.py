"""Positional encoding modules for StackFormer attention blocks.

Provides positional embedding strategies:
- AbsolutePositionEmbedding: learned lookup matrix by position index
- SinusoidalPositionalEmbedding: fixed deterministic sine/cosine frequency basis
- RoPE: Rotary Position Embedding in complex space
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn


class AbsolutePositionEmbedding(nn.Module):
    """Learned absolute positional embedding table.

    Constructor args:
        seq_len (int): Maximum supported sequence position index.
        embed_dim (int): Positional embedding dimension ``C``.
        device (torch.device | str | None, default=None): Target compute device.
        dtype (torch.dtype | None, default=None): Target data type.

    Learnable parameters:
        - embedding.weight: Learned positional embeddings matrix of shape ``(seq_len, embed_dim)``.

    Forward args:
        x (torch.Tensor): Input tensor of shape ``(B, T, C)`` or token sequence.

    Returns:
        torch.Tensor: Positional embeddings tensor of shape ``(B, T, C)``.

    Example:
        >>> pos_emb = AbsolutePositionEmbedding(seq_len=512, embed_dim=128)
        >>> x = torch.randn(4, 32, 128)
        >>> pos = pos_emb(x)
        >>> pos.shape
        torch.Size([4, 32, 128])
    """

    def __init__(
        self,
        seq_len: int,
        embed_dim: int,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.embed_dim = embed_dim
        factory_kwargs = {"device": device, "dtype": dtype}
        self.embedding = nn.Embedding(seq_len, embed_dim, **factory_kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C)
        batch_size, seq_len = x.shape[0], x.shape[1]
        if seq_len > self.seq_len:
            raise ValueError(
                f"Sequence length {seq_len} exceeds maximum {self.seq_len}."
            )
        positions = torch.arange(
            seq_len, device=self.embedding.weight.device, dtype=torch.long
        )
        abs_pos = self.embedding(positions)  # (T, C)
        out = abs_pos.unsqueeze(0).expand(batch_size, seq_len, -1)  # (B, T, C)
        return out


class SinusoidalPositionalEmbedding(nn.Module):
    """Deterministic sinusoidal positional encoding.

    Formula:
        PE[p, 2i]   = sin(p / 10000^(2i/C))
        PE[p, 2i+1] = cos(p / 10000^(2i/C))

    Constructor args:
        seq_len (int): Maximum precomputed sequence length.
        embed_dim (int): Positional embedding dimension ``C`` (must be even).
        device (torch.device | str | None, default=None): Target compute device.
        dtype (torch.dtype | None, default=None): Target data type.

    Forward args:
        x (torch.Tensor): Input sequence tensor of shape ``(B, T, C)``.

    Returns:
        torch.Tensor: Sinusoidal positional embeddings tensor of shape ``(B, T, C)``.

    Example:
        >>> pos_emb = SinusoidalPositionalEmbedding(seq_len=512, embed_dim=128)
        >>> x = torch.randn(4, 32, 128)
        >>> pos = pos_emb(x)
        >>> pos.shape
        torch.Size([4, 32, 128])
    """

    def __init__(
        self,
        seq_len: int,
        embed_dim: int,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if embed_dim % 2 != 0:
            raise ValueError(f"embed_dim must be even, got {embed_dim}")

        factory_kwargs = {"device": device, "dtype": dtype}

        position = torch.arange(seq_len, **factory_kwargs).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2, **factory_kwargs)
            * -(math.log(10000.0) / embed_dim)
        )

        pe = torch.zeros(seq_len, embed_dim, **factory_kwargs)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C)
        B, T = x.shape[:2]
        if T > self.pe.size(0):
            raise ValueError(
                f"Sequence length {T} exceeds maximum {self.pe.size(0)}."
            )
        return self.pe[:T].unsqueeze(0).expand(B, T, -1)  # (B, T, C)


class RoPE(nn.Module):
    """Rotary Position Embedding (RoPE) for attention Query and Key tensors.

    Rotates pairs of hidden dimensions by a position-dependent frequency spectrum,
    directly injecting relative positional information into attention inner products.

    Paper reference:
        RoFormer: Enhanced Transformer with Rotary Position Embedding (Su et al., 2021)
        https://arxiv.org/abs/2104.09864

    Constructor args:
        head_dim (int): Head dimension ``D`` (must be even).
        seq_len (int): Maximum sequence length ``T``.
        device (torch.device | str | None, default=None): Target compute device.
        dtype (torch.dtype | None, default=None): Target data type.
        theta (float, default=10000.0): Base frequency parameter.

    Example:
        >>> rope = RoPE(head_dim=64, seq_len=512)
        >>> q = torch.randn(2, 8, 32, 64)
        >>> q_rot = rope(q)
        >>> q_rot.shape
        torch.Size([2, 8, 32, 64])
    """

    def __init__(
        self,
        head_dim: int,
        seq_len: int,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
        theta: float = 10000.0,
    ) -> None:
        super().__init__()
        self.head_dim = head_dim
        self.seq_len = seq_len
        self.theta = theta
        self.factory_kwargs = {"device": device, "dtype": dtype}

        freq_complex = self._precompute_theta_position_frequency(head_dim, seq_len, theta)
        self.register_buffer("freq_complex", freq_complex, persistent=True)

    def _precompute_theta_position_frequency(
        self, head_dim: int, seq_len: int, theta: float
    ) -> torch.Tensor:
        """Precompute complex rotary frequency spectrum tensor.

        Args:
            head_dim (int): Per-head feature dimension (D).
            seq_len (int): Maximum sequence context length (T).
            theta (float): Base frequency scaling parameter.

        Returns:
            torch.Tensor: Complex frequency tensor of shape ``(T, D // 2)``.
        """
        if head_dim % 2 != 0:
            raise ValueError("head_dim must be even for RoPE")
        dim_half = head_dim // 2
        inv_freq = 1.0 / (
            theta ** (torch.arange(0, dim_half, **self.factory_kwargs) / dim_half)
        )
        pos = torch.arange(seq_len, **self.factory_kwargs)
        freqs = torch.outer(pos, inv_freq)
        freq_complex = torch.polar(torch.ones_like(freqs), freqs)
        return freq_complex  # (seq_len, dim_half)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply Rotary Position Embedding to input Query or Key tensor.

        Args:
            x (torch.Tensor): Input attention tensor of shape ``(B, H, T, D)``.

        Returns:
            torch.Tensor: Rotary position-encoded tensor of shape ``(B, H, T, D)``.
        """
        B, H, T, D = x.shape
        if D % 2 != 0:
            raise ValueError("head_dim must be even for RoPE")

        x_r = x.reshape(B, H, T, D // 2, 2)
        x_complex = torch.view_as_complex(x_r.float())  # (B, H, T, D // 2)

        if T > self.freq_complex.size(0):
            raise ValueError(
                f"Sequence length {T} exceeds maximum {self.freq_complex.size(0)}."
            )
        freqs = self.freq_complex[:T].unsqueeze(0).unsqueeze(0)  # (1, 1, T, D // 2)
        freqs = freqs.to(x_complex)

        x_rot = x_complex * freqs  # (B, H, T, D // 2)

        x_out = torch.view_as_real(x_rot).reshape(B, H, T, D)
        return x_out.to(dtype=x.dtype, device=x.device)  # (B, H, T, D)