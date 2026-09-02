"""Transformer building blocks — from a single block to a full encoder/decoder stack.

Design principles:
  - One config object (BlockConfig) drives everything — no repeated keyword walls.
  - Private factory functions (_build_*) keep each block class clean.
  - EncoderBlock / DecoderBlock own their wiring; stacks own their depth.
  - TransformerBlock is kept for backward compatibility.

Typical usage:
    >>> from stackformer.modules.layer import BlockConfig, TransformerEncoder, TransformerDecoder
    >>> cfg = BlockConfig(embed_dim=512, num_heads=8, ffn="swiglu", attention="gqa", num_kv_heads=2)
    >>> encoder = TransformerEncoder(cfg, num_layers=6, pos_embedding="sinusoidal")
    >>> decoder = TransformerDecoder(cfg, num_layers=6)
    >>> memory = encoder(src)           # (B, S, C)
    >>> out    = decoder(tgt, memory)   # (B, T, C)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import torch
from torch import nn

from stackformer.modules.Attention import (
    Cross_MultiHead_Attention,
    Group_query_Attention,
    Group_query_Attention_With_RoPE,
    Multi_Head_Attention,
    Multi_Head_Attention_With_RoPE,
    Multi_query_Attention,
    Multi_query_Attention_With_RoPE,
    Self_Attention,
)
from stackformer.modules.Feed_forward import (
    FF_GELU,
    FF_GeGLU,
    FF_LeakyReLU,
    FF_ReLU,
    FF_Sigmoid,
    FF_SiLU,
    FF_SwiGLU,
)
from stackformer.modules.Normalization import (
    LayerNorm,
    RMSNorm,
)
# Aliases for backward compat within this file
LayerNormalization = LayerNorm
RMSNormalization = RMSNorm
from stackformer.modules.position_embedding import (
    AbsolutePositionEmbedding,
    SinusoidalPositionalEmbedding,
)


@dataclass
class BlockConfig:
    """Hyperparameter configuration dataclass describing a single transformer block.

    Args:
        embed_dim (int): Model embedding dimension size ``C``.
        num_heads (int): Number of query attention heads ``H``.
        hidden_dim (int, default=0): FFN inner dimension (0 defaults to 4 * embed_dim).
        attention (str, default="mha"): Self-attention variant ("mha", "mha_rope", "mqa", "mqa_rope", "gqa", "gqa_rope", "self").
        num_kv_heads (int | None, default=None): Key/Value heads for GQA variants (defaults to num_heads).
        ffn (str, default="swiglu"): Feed-forward activation type ("relu", "leakyrelu", "gelu", "sigmoid", "silu", "swiglu", "geglu").
        norm (str, default="rmsnorm"): Normalization layer type ("layernorm", "rmsnorm").
        dropout (float, default=0.0): Dropout probability in attention and FFN blocks.
        pre_norm (bool, default=True): If True, use Pre-LN; if False, use Post-LN.
        mask_type (list, default=["causal"]): List of mask pattern names.
        qkv_bias (bool, default=True): Enable bias in Q/K/V linear projections.
        eps (float, default=1e-5): Epsilon for normalization layers.
        device (torch.device | str, default="cpu"): Target compute device.
        dtype (torch.dtype, default=torch.float32): Tensor data type.
    """

    embed_dim: int
    num_heads: int
    hidden_dim: int = 0

    attention: Literal[
        "mha",
        "mha_rope",
        "mqa",
        "mqa_rope",
        "gqa",
        "gqa_rope",
        "self",
    ] = "mha"
    num_kv_heads: int | None = None

    ffn: Literal[
        "relu",
        "leakyrelu",
        "gelu",
        "sigmoid",
        "silu",
        "swiglu",
        "geglu",
    ] = "swiglu"
    norm: Literal["layernorm", "rmsnorm"] = "rmsnorm"

    dropout: float = 0.0
    pre_norm: bool = True

    mask_type: list = field(default_factory=lambda: ["causal"])
    qkv_bias: bool = True

    eps: float = 1e-5

    device: str = "cpu"
    dtype: torch.dtype = torch.float32

    def __post_init__(self) -> None:
        if self.hidden_dim == 0:
            self.hidden_dim = 4 * self.embed_dim

        if self.num_kv_heads is None:
            self.num_kv_heads = self.num_heads

        if self.embed_dim % self.num_heads != 0:
            raise ValueError(
                f"embed_dim ({self.embed_dim}) must be divisible by " f"num_heads ({self.num_heads})"
            )

        if ("gqa" in self.attention or "mqa" in self.attention) and self.num_heads % self.num_kv_heads != 0:
            raise ValueError(
                f"num_heads ({self.num_heads}) must be divisible by "
                f"num_kv_heads ({self.num_kv_heads}) for {self.attention}"
            )


# Private factories
def _build_attention(cfg: BlockConfig) -> nn.Module:
    """Return the self-attention module described by cfg."""
    shared = dict(
        dropout=cfg.dropout,
        mask_type=cfg.mask_type,
        qkv_bias=cfg.qkv_bias,
        device=cfg.device,
        dtype=cfg.dtype,
    )
    embed_dim, num_heads, num_kv_heads = cfg.embed_dim, cfg.num_heads, cfg.num_kv_heads

    key = cfg.attention.lower()
    if key == "self":
        return Self_Attention(embed_dim, **shared)
    elif key == "mha":
        return Multi_Head_Attention(embed_dim, num_heads, **shared)
    elif key == "mha_rope":
        return Multi_Head_Attention_With_RoPE(embed_dim, num_heads, **shared)
    elif key == "mqa":
        return Multi_query_Attention(embed_dim, num_heads, **shared)
    elif key == "mqa_rope":
        return Multi_query_Attention_With_RoPE(embed_dim, num_heads, **shared)
    elif key == "gqa":
        return Group_query_Attention(embed_dim, num_heads, num_kv_heads, **shared)
    elif key == "gqa_rope":
        return Group_query_Attention_With_RoPE(embed_dim, num_heads, num_kv_heads, **shared)
    else:
        raise ValueError(
            f"Unknown attention '{cfg.attention}'. "
            f"Available: ['mha', 'mha_rope', 'mqa', 'mqa_rope', 'gqa', 'gqa_rope', 'self']"
        )

        
def _build_cross_attention(cfg: BlockConfig) -> nn.Module:
    """Return a cross-attention module (queries from target, K/V from memory)."""
    return Cross_MultiHead_Attention(
        cfg.embed_dim, cfg.num_heads,
        dropout  = cfg.dropout,
        qkv_bias = cfg.qkv_bias,
        device   = cfg.device,
        dtype    = cfg.dtype,
    )


def _build_ffn(cfg: BlockConfig) -> nn.Module:
    """Return the FFN module described by *cfg*."""
    hw = dict(device=cfg.device, dtype=cfg.dtype)
    E, M, D = cfg.embed_dim, cfg.hidden_dim, cfg.dropout
    
    key = cfg.ffn.lower()
    if key == "relu":
        return FF_ReLU(E, M, D, **hw)
    elif key == "leakyrelu":
        return FF_LeakyReLU(E, M, D, **hw)
    elif key == "gelu":
        return FF_GELU(E, M, D, **hw)
    elif key == "sigmoid":
        return FF_Sigmoid(E, M, D, **hw)
    elif key == "silu":
        return FF_SiLU(E, M, D, **hw)
    elif key == "swiglu":
        return FF_SwiGLU(E, M, D, **hw)
    elif key == "geglu":
        return FF_GeGLU(E, M, D, **hw)
    else:
        raise ValueError(
            f"Unknown FFN '{cfg.ffn}'. "
            f"Available: ['relu', 'leakyrelu', 'gelu', 'sigmoid', 'silu', 'swiglu', 'geglu']"
        )

def _build_norm(cfg: BlockConfig) -> nn.Module:
    """Return one normalization layer described by *cfg*."""
    hw = dict(device=cfg.device, dtype=cfg.dtype)

    key = cfg.norm.lower()
    if key == "layernorm":
        return LayerNormalization(cfg.embed_dim, eps=cfg.eps, **hw)
    elif key == "rmsnorm":
        return RMSNormalization(cfg.embed_dim, eps=cfg.eps, **hw)
    else:
        raise ValueError(
            f"Unknown norm '{cfg.norm}'. "
            f"Available: ['layernorm', 'rmsnorm']"
        )

def _build_pos_embedding(
    name: str | None,
    max_seq_len: int,
    cfg: BlockConfig,
) -> nn.Module | None:
    """Return a positional embedding module, or None.

    Note: RoPE positional encoding is handled *inside* the attention
    modules when using ``attention='mha_rope'`` / ``'gqa_rope'`` etc.
    Do not add a separate pos_embedding in that case.
    """
    if name is None:
        return None

    hw = dict(device=cfg.device, dtype=cfg.dtype)

    registry: dict[str, nn.Module] = {
        "sinusoidal": SinusoidalPositionalEmbedding(max_seq_len, cfg.embed_dim, **hw),
        "absolute":   AbsolutePositionEmbedding(max_seq_len, cfg.embed_dim, **hw),
    }
    key = name.lower()
    if key not in registry:
        raise ValueError(
            f"Unknown positional embedding '{name}'. "
            f"Available: {list(registry)}"
        )
    return registry[key]


# Blocks
class EncoderBlock(nn.Module):
    """One transformer encoder layer containing self-attention and FFN sub-layers.

    Wiring (pre-norm):
        x -> norm1 -> self_attn -> + -> norm2 -> ffn -> + -> output
        |__________________________| |__________________|

    Wiring (post-norm):
        x -> self_attn -> + -> norm1 -> ffn -> + -> norm2 -> output
        |_________________| |__________________|

    Constructor args:
        cfg (BlockConfig): Block configuration dataclass.

    Forward args:
        x (torch.Tensor): Input sequence tensor of shape ``(B, T, C)``.
        mask (bool, default=False): Apply attention mask (default False for encoder).

    Returns:
        torch.Tensor: Encoder block output tensor of shape ``(B, T, C)``.

    Example:
        >>> cfg = BlockConfig(embed_dim=512, num_heads=8, ffn="swiglu")
        >>> block = EncoderBlock(cfg)
        >>> y = block(torch.randn(2, 64, 512))
        >>> y.shape
        torch.Size([2, 64, 512])
    """


    def __init__(self, cfg: BlockConfig) -> None:
        super().__init__()
        self.pre_norm = cfg.pre_norm
        self.self_attn = _build_attention(cfg)
        self.ffn = _build_ffn(cfg)
        self.norm1 = _build_norm(cfg)
        self.norm2 = _build_norm(cfg)

    def forward(self, x: torch.Tensor, mask: bool = False) -> torch.Tensor:
        # x: (B, T, C)
        if self.pre_norm:
            x = x + self.self_attn(self.norm1(x), mask=mask)
            x = x + self.ffn(self.norm2(x))
        else:  # post-norm
            x = self.norm1(x + self.self_attn(x, mask=mask))
            x = self.norm2(x + self.ffn(x))
        return x  # (B, T, C)


class DecoderBlock(nn.Module):
    """One transformer decoder layer containing self-attention, cross-attention, and FFN sub-layers.

    Wiring (pre-norm):
        x -> norm1 -> self_attn (causal) -> + -> norm2 -> cross_attn(memory) -> + -> norm3 -> ffn -> + -> output
        |________________________________| |_________________________________| |_______________|

    Constructor args:
        cfg (BlockConfig): Block configuration dataclass.

    Forward args:
        x (torch.Tensor): Target sequence input tensor of shape ``(B, T, C)``.
        memory (torch.Tensor): Encoder memory sequence tensor of shape ``(B, S, C)``.
        self_mask (bool, default=True): Apply causal mask to self-attention.
        cross_mask (bool, default=False): Apply mask to cross-attention.
        cross_attn_mask (torch.Tensor | None, default=None): Explicit cross-attention mask tensor.

    Returns:
        torch.Tensor: Decoder block output tensor of shape ``(B, T, C)``.

    Example:
        >>> cfg = BlockConfig(embed_dim=512, num_heads=8)
        >>> block = DecoderBlock(cfg)
        >>> tgt, memory = torch.randn(2, 30, 512), torch.randn(2, 40, 512)
        >>> y = block(tgt, memory)
        >>> y.shape
        torch.Size([2, 30, 512])
    """

    def __init__(self, cfg: BlockConfig) -> None:
        super().__init__()
        self.pre_norm = cfg.pre_norm
        self.self_attn = _build_attention(cfg)
        self.cross_attn = _build_cross_attention(cfg)
        self.ffn = _build_ffn(cfg)
        self.norm1 = _build_norm(cfg)
        self.norm2 = _build_norm(cfg)
        self.norm3 = _build_norm(cfg)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        self_mask: bool = True,
        cross_mask: bool = False,
        cross_attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # x: (B, T, C), memory: (B, S, C)
        if self.pre_norm:
            x = x + self.self_attn(self.norm1(x), mask=self_mask)
            x = x + self.cross_attn(
                self.norm2(x), memory, mask=cross_mask, attn_mask=cross_attn_mask
            )
            x = x + self.ffn(self.norm3(x))
        else:  # post-norm
            x = self.norm1(x + self.self_attn(x, mask=self_mask))
            x = self.norm2(
                x + self.cross_attn(x, memory, mask=cross_mask, attn_mask=cross_attn_mask)
            )
            x = self.norm3(x + self.ffn(x))
        return x  # (B, T, C)


class TransformerEncoder(nn.Module):
    """A stacked sequence of EncoderBlock layers.

    Constructor args:
        cfg (BlockConfig): Shared block configuration.
        num_layers (int): Number of encoder layers to stack.
        pos_embedding (str | None, default=None): Positional embedding type ("sinusoidal", "absolute", or None).
        max_seq_len (int, default=4096): Maximum sequence length for positional embeddings.

    Forward args:
        x (torch.Tensor): Token embedding input tensor of shape ``(B, T, C)``.
        mask (bool, default=False): Masking flag forwarded to each block.

    Returns:
        torch.Tensor: Encoder stack output representations tensor of shape ``(B, T, C)``.

    Example:
        >>> cfg = BlockConfig(embed_dim=512, num_heads=8, ffn="swiglu", attention="gqa", num_kv_heads=2)
        >>> encoder = TransformerEncoder(cfg, num_layers=6, pos_embedding="sinusoidal")
        >>> memory = encoder(torch.randn(2, 64, 512))
        >>> memory.shape
        torch.Size([2, 64, 512])
    """

    def __init__(
        self,
        cfg: BlockConfig,
        num_layers: int,
        pos_embedding: str | None = None,
        max_seq_len: int = 4096,
    ) -> None:
        super().__init__()
        self.pos_embedding = _build_pos_embedding(pos_embedding, max_seq_len, cfg)
        self.layers = nn.ModuleList([EncoderBlock(cfg) for _ in range(num_layers)])
        self.final_norm = _build_norm(cfg) if cfg.pre_norm else nn.Identity()

    def forward(self, x: torch.Tensor, mask: bool = False) -> torch.Tensor:
        # x: (B, T, C)
        if self.pos_embedding is not None:
            x = x + self.pos_embedding(x)
        for layer in self.layers:
            x = layer(x, mask=mask)
        return self.final_norm(x)  # (B, T, C)


class TransformerDecoder(nn.Module):
    """A stacked sequence of DecoderBlock layers.

    Constructor args:
        cfg (BlockConfig): Shared block configuration.
        num_layers (int): Number of decoder layers to stack.
        pos_embedding (str | None, default=None): Positional embedding type ("sinusoidal", "absolute", or None).
        max_seq_len (int, default=4096): Maximum sequence length for positional embeddings.

    Forward args:
        x (torch.Tensor): Target token embedding tensor of shape ``(B, T, C)``.
        memory (torch.Tensor): Encoder output memory tensor of shape ``(B, S, C)``.
        self_mask (bool, default=True): Causal self-attention mask flag.
        cross_mask (bool, default=False): Cross-attention mask flag.
        cross_attn_mask (torch.Tensor | None, default=None): Explicit cross-attention mask tensor.

    Returns:
        torch.Tensor: Decoder stack output tensor of shape ``(B, T, C)``.

    Example:
        >>> cfg = BlockConfig(embed_dim=512, num_heads=8)
        >>> decoder = TransformerDecoder(cfg, num_layers=6)
        >>> out = decoder(torch.randn(2, 30, 512), torch.randn(2, 40, 512))
        >>> out.shape
        torch.Size([2, 30, 512])
    """

    def __init__(
        self,
        cfg: BlockConfig,
        num_layers: int,
        pos_embedding: str | None = None,
        max_seq_len: int = 4096,
    ) -> None:
        super().__init__()
        self.pos_embedding = _build_pos_embedding(pos_embedding, max_seq_len, cfg)
        self.layers = nn.ModuleList([DecoderBlock(cfg) for _ in range(num_layers)])
        self.final_norm = _build_norm(cfg) if cfg.pre_norm else nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        self_mask: bool = True,
        cross_mask: bool = False,
        cross_attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # x: (B, T, C), memory: (B, S, C)
        if self.pos_embedding is not None:
            x = x + self.pos_embedding(x)
        for layer in self.layers:
            x = layer(
                x,
                memory,
                self_mask=self_mask,
                cross_mask=cross_mask,
                cross_attn_mask=cross_attn_mask,
            )
        return self.final_norm(x)  # (B, T, C)


class TransformerBlock(nn.Module):
    """Single encoder block alias (retained for backward compatibility).

    Constructor args:
        embed_dim (int): Model embedding dimension size ``C``.
        num_heads (int): Number of attention heads ``H``.
        hidden_dim (int): FFN inner dimension size ``M``.
        attention (str, default="mha"): Self-attention type.
        num_kv_heads (int | None, default=None): Key/Value heads for GQA.
        ffn (str, default="relu"): FFN activation type.
        norm (str, default="layernorm"): Normalization layer type.
        dropout (float, default=0.0): Dropout probability.
        pre_norm (bool, default=True): Use Pre-LN if True, Post-LN if False.
        eps (float, default=1e-5): Epsilon for normalization.
        device (torch.device | str, default="cpu"): Target device.
        dtype (torch.dtype, default=torch.float32): Tensor data type.

    Forward args:
        x (torch.Tensor): Input sequence tensor of shape ``(B, T, C)``.
        mask (bool, default=True): Apply causal mask.

    Returns:
        torch.Tensor: Output tensor of shape ``(B, T, C)``.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        hidden_dim: int,
        attention: str = "mha",
        num_kv_heads: int | None = None,
        ffn: str = "relu",
        norm: str = "layernorm",
        dropout: float = 0.0,
        pre_norm: bool = True,
        eps: float = 1e-5,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        cfg = BlockConfig(
            embed_dim=embed_dim,
            num_heads=num_heads,
            hidden_dim=hidden_dim,
            attention=attention,
            num_kv_heads=num_kv_heads,
            ffn=ffn,
            norm=norm,
            dropout=dropout,
            pre_norm=pre_norm,
            eps=eps,
            device=device,
            dtype=dtype,
        )
        self._block = EncoderBlock(cfg)

    def forward(self, x: torch.Tensor, mask: bool = True) -> torch.Tensor:
        # x: (B, T, C)
        return self._block(x, mask=mask)