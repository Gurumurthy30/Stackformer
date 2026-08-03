"""Attention implementations used by Stackformer.

This module provides a research-to-production set of attention operators:
standard self-attention, RoPE variants, cross-attention, MQA/GQA, local-window
attention, and KV-cache inference attention.

Core equation used by almost all classes:

    Attention(Q, K, V) = softmax((Q K^T) / sqrt(d_k) + M) V

where ``M`` is usually a causal mask (``-inf`` on disallowed positions).

Notation:
- ``B``: batch size
- ``T``: query sequence length (current tokens)
- ``S``: key/value sequence length (context or cache length)
- ``C``: embedding dimension
- ``H``: number of query heads
- ``H_kv``: number of key/value heads (MQA: 1, GQA: 1 < H_kv < H, MHA: H_kv == H)
- ``D``: head dimension, usually ``C // H``

Implementation notes:
- Inputs are moved to ``device``/``dtype`` configured in each module.
- Masks are cached by (mask names, sequence shape, backend-relevant params)
  to reduce allocation/compilation overhead.
- RoPE modules require even head dimension (enforced with assertions).
- MQA/GQA classes (``Multi_query_Attention*``, ``Group_query_Attention*``)
  support a ``backend`` switch between ``"sdpa"`` and ``"flex"``. Both
  kernels understand ``enable_gqa=True``, which lets them broadcast K/V from
  ``H_kv`` heads up to ``H`` heads internally. Because of this, these classes
  keep K/V at their native, smaller head count and never materialize a
  repeated/expanded copy -- faster and lower memory than the classic
  ``.expand()`` / ``.repeat_interleave()`` approach.

Quick start:
    >>> import torch
    >>> from stackformer.modules.Attention import Multi_Head_Attention
    >>> x = torch.randn(2, 32, 256)
    >>> attn = Multi_Head_Attention(embed_dim=256, num_heads=8)
    >>> y = attn(x, mask=True)
    >>> y.shape
    torch.Size([2, 32, 256])

    >>> from stackformer.modules.Attention import Group_query_Attention
    >>> gqa = Group_query_Attention(embed_dim=256, num_query_heads=8, num_kv_heads=2,
    ...                              mask_type="causal", backend="flex")
    >>> y = gqa(torch.randn(2, 32, 256))
"""

from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from stackformer.modules.masks import _get_attention_mask, _get_block_mask
from stackformer.modules.attention_engine import _run_sdpa, _run_flex_attention
from stackformer.cache import StaticKVCache


_ROPE_FREQ_CACHE: dict[tuple[int, int, str, torch.dtype, float], torch.Tensor] = {}


def _build_rope_frequency(
    head_dim: int, seq_len: int, device: torch.device | str, dtype: torch.dtype, theta: float = 10000.0
) -> torch.Tensor:
    """Precompute complex rotary positional frequency spectrum for RoPE.

    Args:
        head_dim (int): Dimension per attention head (must be even).
        seq_len (int): Maximum sequence length to compute frequencies for.
        device (torch.device | str): Target compute device.
        dtype (torch.dtype): Target data type.
        theta (float, default=10000.0): RoPE base frequency multiplier.

    Returns:
        torch.Tensor: Complex frequency tensor of shape ``(seq_len, head_dim // 2)``.
    """
    assert head_dim % 2 == 0, "head_dim must be even for RoPE"
    key = (head_dim, seq_len, str(device), dtype, theta)
    if key in _ROPE_FREQ_CACHE:
        return _ROPE_FREQ_CACHE[key]

    dim_half = head_dim // 2
    inv_freq = 1.0 / (theta ** (torch.arange(0, dim_half, device=device, dtype=torch.float32) / dim_half))
    pos = torch.arange(seq_len, device=device, dtype=torch.float32)
    freqs = torch.outer(pos, inv_freq)
    freq_complex = torch.polar(torch.ones_like(freqs), freqs)
    _ROPE_FREQ_CACHE[key] = freq_complex
    return freq_complex


def _apply_rotary_position_embedding(x: torch.Tensor, freq_complex: torch.Tensor) -> torch.Tensor:
    """Apply Rotary Position Embeddings (RoPE) via complex multiplication.

    Args:
        x (torch.Tensor): Attention Q or K tensor of shape ``(B, H, T, D)``.
        freq_complex (torch.Tensor): Complex frequency tensor slice of shape ``(T, D // 2)``.

    Returns:
        torch.Tensor: Position-encoded Q or K tensor of shape ``(B, H, T, D)``.
    """
    B, H, T, D = x.shape
    assert D % 2 == 0, "head_dim must be even for RoPE"

    x = x.view(B, H, T, D // 2, 2)
    x_complex = torch.view_as_complex(x)  # (B, H, T, D//2)

    freq = freq_complex[:T].unsqueeze(0).unsqueeze(0)  # (1, 1, T, D//2)
    x_rot = x_complex * freq  # rotate via complex mult

    x_out = torch.view_as_real(x_rot).view(B, H, T, D)
    return x_out.to(dtype=x.dtype, device=x.device)


class Self_Attention(nn.Module):
    """Single-head causal/self attention.

    Mathematical form:
        - Q = X W_q, K = X W_k, V = X W_v
        - A = softmax((Q K^T) / sqrt(C) + M)
        - Y = A V W_o

    Constructor args:
        embed_dim (int): Input/hidden size ``C``.
        dropout (float, default=0.0): Dropout probability on attention probabilities after softmax.
        mask_type (list[str] | None, default=None): Masking type ('causal', 'sliding_window').
        qkv_bias (bool, default=False): Enables bias terms in Q/K/V projection layers.
        device (torch.device | str, default='cpu'): Parameter and compute device.
        dtype (torch.dtype, default=torch.float32): Parameter and compute dtype.

    Forward args:
        x (torch.Tensor): Input sequence tensor of shape ``(B, T, C)``.
        mask (bool, default=True): Apply causal masking.

    Returns:
        torch.Tensor: Output tensor of shape ``(B, T, C)``.

    Example:
        >>> layer = Self_Attention(embed_dim=64, dropout=0.0)
        >>> x = torch.randn(4, 32, 64)
        >>> y = layer(x, mask=True)
    """

    def __init__(
        self,
        embed_dim: int,
        dropout: float = 0.0,
        mask_type: list[str] | None = None,
        qkv_bias: bool = False,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
        **mask_kwargs,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.device = device
        self.dtype = dtype
        self.mask_type = mask_type
        self.mask_kwargs = mask_kwargs

        self.qkv_proj = nn.Linear(embed_dim, embed_dim * 3, bias=qkv_bias, device=device, dtype=dtype)
        self.out_proj = nn.Linear(embed_dim, embed_dim, device=device, dtype=dtype)
        self.dropout_p = dropout
        self._causal_mask_cache = OrderedDict()


    def _get_or_create_mask(self, seq_len: int, device):
        return _get_attention_mask(
            self._causal_mask_cache,
            self.mask_type,
            seq_len,
            device,
            **self.mask_kwargs,
        )

    def forward(self, x, mask=True):
        B, T, C = x.shape
        
        # Project Q, K, V
        qkv = self.qkv_proj(x)                      # (B, T, 3*C)
        q, k, v = qkv.split(self.embed_dim, dim=-1)    # each (B, T, C)

        # Add single head dimension for SDPA
        q = q.unsqueeze(1)  # (B, 1, T, C)
        k = k.unsqueeze(1)
        v = v.unsqueeze(1)

        attn_mask = None
        if mask:
            attn_mask = self._get_or_create_mask(seq_len=T, device=x.device)

        out = _run_sdpa(
            q,k,v,
            attn_mask = attn_mask,
            dropout_p=self.dropout_p)

        # Remove head dimension
        out = out.squeeze(1)  # (B, T, C)

        return self.out_proj(out)

class Multi_Head_Attention(nn.Module):
    """Standard multi-head self-attention (MHA).

    Why/when to use:
    - Baseline attention for encoder and decoder blocks.
    - Multiple heads learn different relation subspaces.

    Constructor args:
        embed_dim (int, required): Model width ``C``.
        num_heads (int, required): Number of query heads ``H``.
            Rule: ``embed_dim % num_heads == 0`` (enforced).
        dropout (float, optional, default=0.0): Dropout on attention probs.
        qkv_bias (bool, optional, default=False): Bias in Q/K/V projections.
        device (str or torch.device, optional, default='cpu').
        dtype (torch.dtype, optional, default=torch.float32).
        mask_type ([str], optional, default=['causal']): 'causal' or 'sliding_window'.
        
    Forward args:
        x (torch.Tensor): ``(B, T, C)``.
        mask (bool, optional, default=True): Apply causal mask.

    Returns:
        torch.Tensor: ``(B, T, C)``.

    Complexity:
        Time/memory are O(B * H * T^2 * D), dominated by attention matrix.

    Example:
        >>> layer = Multi_Head_Attention(embed_dim=512, num_heads=8)
        >>> y = layer(torch.randn(2, 128, 512), mask=True)
    """
    def __init__(self, embed_dim, num_heads, dropout=0.0, mask_type=None,
                 qkv_bias=False,device='cpu', dtype=torch.float32, 
                 **mask_kwargs):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads  # Each head gets a slice of the embedding
        
        self.mask_type = mask_type
        self.device = device
        self.dtype = dtype
        self.mask_kwargs = mask_kwargs

        # Linear layer
        self.qkv_proj = nn.Linear(embed_dim, embed_dim*3, bias=qkv_bias, device=device, dtype=self.dtype)
        
        # Final output projection after attention
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=qkv_bias, device=device, dtype=self.dtype)

        # Dropout applied to the attention weights
        self.dropout_p = dropout
        
        # Cache for causal masks keyed by sequence length
        self._causal_mask_cache = OrderedDict()

    def _get_or_create_mask(self, seq_len: int, device):
        return _get_attention_mask(
            self._causal_mask_cache,
            self.mask_type,
            seq_len,
            device,
            **self.mask_kwargs,
        )

    def forward(self, x, mask=True):
        B, T, C = x.shape
        
        qkv = self.qkv_proj(x)                      # (B, T, 3*C)
        q, k, v = qkv.split(self.embed_dim, dim=-1)    # each (B, T, C)
        
        # Reshape for multi-head attention:
        # (B, T, C) → (B, T, num_heads, head_dim) → (B, num_heads, T, head_dim)
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        
        causal_mask = None
        # Apply causal mask if needed
        if mask:
            causal_mask = self._get_or_create_mask(seq_len=T, device=x.device)  # (T, T)

        out = _run_sdpa(
            q, k, v,
            attn_mask = causal_mask,
            dropout_p=self.dropout_p
        )
        
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        
        # Final linear projection
        return self.out_proj(out)  # (B, T, C)

class Multi_Head_Attention_With_RoPE(nn.Module):
    """Multi-head self-attention with Rotary Positional Embedding (RoPE).

    RoPE applies a position-dependent 2D rotation on every pair of query/key
    channels, injecting relative position directly into dot products.

    Constructor args:
        embed_dim (int, required).
        num_heads (int, required).
            Rules:
            - ``embed_dim % num_heads == 0``
            - ``head_dim`` must be even for RoPE pair-rotation.
        dropout (float, optional, default=0.0).
        qkv_bias (bool, optional, default=False).
        device (optional, default='cpu').
        dtype (optional, default=torch.float32).
        mask_type ([str], optional, default=['causal']): 'causal' or 'sliding_window'.

    Forward args:
        x (torch.Tensor): ``(B, T, C)``.
        mask (bool, optional, default=True).

    Returns:
        torch.Tensor: ``(B, T, C)``.
    """
    def __init__(self, embed_dim, num_heads, dropout=0.0, mask_type=None, qkv_bias=False,device='cpu', dtype=torch.float32, **mask_kwargs):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.mask_type = mask_type
        self.head_dim = embed_dim // num_heads  # Each head gets a slice of the embedding
        self.device = device
        self.dtype = dtype
        self.mask_kwargs = mask_kwargs

        # Linear layer
        self.qkv_proj = nn.Linear(embed_dim, embed_dim * 3, bias=qkv_bias, device=device, dtype=self.dtype)
        
        # Final output projection after attention
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=qkv_bias, device=device, dtype=self.dtype)

        # Dropout applied to the attention weights
        self.dropout_p = dropout
        
        # Cache for causal masks keyed by sequence length
        self._causal_mask_cache = OrderedDict()

    def _get_or_create_mask(self, seq_len: int, device):
        return _get_attention_mask(
            self._causal_mask_cache,
            self.mask_type,
            seq_len,
            device,
            **self.mask_kwargs,
        )
    
    def _precompute_theta_position_frequency(self, head_dim: int, seq_len: int, device: torch.device,theta: float = 10000.0):
        return _build_rope_frequency(head_dim, seq_len, device, self.dtype, theta=theta)

    def forward(self, x, mask=True, theta: float=10000.0):
        B, T, C = x.shape

        qkv = self.qkv_proj(x)  # (B, T, 3*C)
        q, k, v = qkv.split(self.embed_dim, dim=-1)  # each (B, T, C)

        # Reshape for multi-head attention:
        # (B, T, C) → (B, T, num_heads, head_dim) → (B, num_heads, T, head_dim)
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        
        freq = self._precompute_theta_position_frequency(self.head_dim, T, device=x.device,theta=theta)
        q = _apply_rotary_position_embedding(q, freq)
        k = _apply_rotary_position_embedding(k, freq)
        
        causal_mask = None
        # Apply causal mask if needed
        if mask:
            causal_mask = self._get_or_create_mask(seq_len=T, device=x.device)  # (T, T)

        out = _run_sdpa(
            q, k, v,
            attn_mask = causal_mask,
            dropout_p=self.dropout_p
        )
        
        out = out.transpose(1, 2).contiguous().view(B, T, C)

        # Final linear projection
        return self.out_proj(out)  # (B, T, C)

class Cross_MultiHead_Attention(nn.Module):
    """Cross-attention: queries from ``x``, keys/values from ``context``.

    Constructor args:
        embed_dim (int, required).
        num_heads (int, required): ``embed_dim % num_heads == 0``.
        dropout (float, optional, default=0.0).
        qkv_bias (bool, optional, default=False).
        device (optional, default='cpu').
        dtype (optional, default=torch.float32).
        mask_type ([str], optional, default=['causal']): 'causal' or 'sliding_window'.

    Forward args:
        x (torch.Tensor): Query tensor ``(B, T, C)``.
        context (torch.Tensor): Key/value tensor ``(B, S, C)``.
        mask (bool, optional, default=True): Applies causal mask only when
            ``T == S`` in this implementation.

    Returns:
        torch.Tensor: ``(B, T, C)``.
    """
    def __init__(self, embed_dim, num_heads, dropout=0.0, mask_type=None, qkv_bias=False,device='cpu', dtype=torch.float32, **mask_kwargs):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.mask_type = mask_type
        self.device = device
        self.dtype = dtype
        self.mask_kwargs = mask_kwargs

        # Linear layer
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=qkv_bias, device=device, dtype=dtype)
        self.kv_proj = nn.Linear(embed_dim, embed_dim * 2, bias=qkv_bias, device=device, dtype=dtype)

        # Final output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=qkv_bias, device=device, dtype=dtype)
        
        # Dropout applied to the attention weights
        self.dropout_p = dropout
        
        # Cache for causal masks (optional)
        self._causal_mask_cache = OrderedDict()

    def _get_or_create_mask(self, seq_len: int, device):
        return _get_attention_mask(
            self._causal_mask_cache,
            self.mask_type,
            seq_len,
            device,
            **self.mask_kwargs,
        )

    def forward(self, x, context, mask=False, attn_mask: torch.Tensor | None = None):
        B, T, C = x.shape
        S = context.size(1)  # (B, S, C)

        # Compute Q from x, and K/V from context
        q = self.q_proj(x)  # (B, T, C)
        kv = self.kv_proj(context)
        k, v = kv.split(self.embed_dim, dim=-1)  # each (B, S, C)

        # Reshape to multi-head format
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)  # (B, num_heads, T, head_dim)
        k = k.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)  # (B, num_heads, S, head_dim)
        v = v.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)  # (B, num_heads, S, head_dim)

        if attn_mask is not None:
            if attn_mask.dim() == 2 and attn_mask.shape != (T, S):
                raise ValueError(f"Cross attention mask must have shape (T, S)=({T}, {S}); got {tuple(attn_mask.shape)}")
            causal_mask = attn_mask
        elif mask:
            if T != S:
                raise ValueError("Causal cross-attention mask requires T == S. Provide explicit attn_mask with shape (T, S).")
            causal_mask = self._get_or_create_mask(seq_len=T, device=x.device)
        else:
            causal_mask = None

        out = _run_sdpa(
            q, k, v,
            attn_mask = causal_mask,
            dropout_p=self.dropout_p
        )
        
        out = out.transpose(1, 2).contiguous().view(B, T, C)

        # Final output projection
        return self.out_proj(out)
    
class Multi_query_Attention(nn.Module):
    """Multi-Query Attention (MQA): many query heads, one shared K/V head.

    Backend note:
        K/V are kept at a single head, shape ``(B, 1, T, D)`` -- they are
        **not** manually broadcast to ``num_heads`` heads. Instead the
        attention kernel is called with ``enable_gqa=True``, which both SDPA
        (torch >= 2.5) and FlexAttention understand natively: they repeat K/V
        across query-head groups internally, on the fly, without allocating
        an expanded copy. This is strictly faster and lighter than the old
        ``.expand()`` approach, especially as ``num_heads`` grows.

    Constructor args:
        embed_dim (int, required).
        num_heads (int, required): Number of query heads. Rule:
            ``embed_dim % num_heads == 0``.
        dropout (float, optional, default=0.0): SDPA-only; FlexAttention has
            no built-in dropout, so this is ignored when ``backend="flex"``.
        qkv_bias (bool, optional, default=False).
        device (optional, default='cpu').
        dtype (optional, default=torch.float32).
        backend (Literal["sdpa", "flex"], optional, default="sdpa"): Which
            attention kernel to run.
        combine (Literal["or", "and"], optional, default="or"): How multiple
            mask types in ``mask_type`` combine.
        mask_type ([str], optional, default=['causal']): 'causal' or 'sliding_window'.

    Forward args:
        x (torch.Tensor): ``(B, T, C)``.
        mask (bool, optional, default=True).

    Returns:
        torch.Tensor: ``(B, T, C)``.

    Example:
        >>> mqa = Multi_query_Attention(embed_dim=512, num_heads=8, mask_type="causal")
        >>> y = mqa(torch.randn(2, 128, 512))
        >>> mqa_flex = Multi_query_Attention(embed_dim=512, num_heads=8,
        ...                                   mask_type="causal", backend="flex")
    """
    def __init__(self, embed_dim, num_heads, dropout=0.0, mask_type=None, qkv_bias=False,
                 device='cpu', dtype=torch.float32, backend: Literal["sdpa", "flex"] = "sdpa",
                 combine: Literal["or", "and"] = "or", **mask_kwargs):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        if backend not in ("sdpa", "flex"):
            raise ValueError(f"backend must be 'sdpa' or 'flex', got {backend!r}")

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.mask_type = mask_type
        self.combine = combine
        self.backend = backend
        self.device = device
        self.dtype = dtype
        self.mask_kwargs = mask_kwargs

        # Projection layers
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=qkv_bias, device=device, dtype=self.dtype)
        self.kv_proj = nn.Linear(embed_dim, self.head_dim * 2, bias=qkv_bias, device=device, dtype=self.dtype)
        
        # Output final projection
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=qkv_bias, device=device, dtype=self.dtype)
        
        # Dropout applied to the attention weights (sdpa backend only)
        self.dropout_p = dropout
        
        # Cache holds dense masks (sdpa) or BlockMasks (flex) -- never both
        # at once, since backend is fixed for the lifetime of this module.
        self._causal_mask_cache = OrderedDict()

    def _get_or_create_mask(self, seq_len: int, device):
        if self.backend == "sdpa":
            return _get_attention_mask(
                self._causal_mask_cache, self.mask_type, seq_len, device,
                combine=self.combine, **self.mask_kwargs,
            )
        return _get_block_mask(
            self._causal_mask_cache, self.mask_type, seq_len, device,
            combine=self.combine, **self.mask_kwargs,
        )

    def forward(self, x, mask=True):
        B, T, C = x.shape

        # Project
        q = self.q_proj(x)                    # (B, T, C)
        kv = self.kv_proj(x)                  # (B, T, 2*D)
        k, v = kv.split(self.head_dim, dim=-1) # (B, T, D) each

        # Multi-head queries
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2) # (B, H, T, D)
        # Single shared K/V head -- left at (B, 1, T, D). enable_gqa below
        # does the broadcast to H heads inside the kernel; no .expand() here.
        k = k.unsqueeze(1)                    # (B, 1, T, D)
        v = v.unsqueeze(1)                    # (B, 1, T, D)

        attn_mask = self._get_or_create_mask(seq_len=T, device=x.device) if mask else None

        if self.backend == "sdpa":
            out = _run_sdpa(q, k, v, attn_mask=attn_mask, dropout_p=self.dropout_p, enable_gqa=True)
        else:
            out = _run_flex_attention(q, k, v, block_mask=attn_mask, enable_gqa=True)

        out = out.transpose(1, 2).reshape(B, T, C)

        return self.out_proj(out)
    
class Multi_query_Attention_With_RoPE(nn.Module):
    """MQA with RoPE on queries and the shared key.

    Backend note:
        Same ``enable_gqa=True`` treatment as ``Multi_query_Attention``: K/V
        stay at a single head and are never expanded. RoPE is applied to the
        single-head ``k`` exactly as it would be to any other head count --
        rotation is per-position, per-head-independent, so it composes fine
        with the un-expanded shape.

    Constructor args:
        embed_dim (int, required).
        num_heads (int, required): ``embed_dim % num_heads == 0`` and even
            ``head_dim`` for RoPE.
        dropout (float, optional, default=0.0): SDPA-only.
        qkv_bias (bool, optional, default=False).
        device (optional, default='cpu').
        dtype (optional, default=torch.float32).
        backend (Literal["sdpa", "flex"], optional, default="sdpa").
        combine (Literal["or", "and"], optional, default="or").
        mask_type ([str], optional, default=['causal']): 'causal' or 'sliding_window'.

    Forward args:
        x (torch.Tensor): ``(B, T, C)``.
        mask (bool, optional, default=True).

    Returns:
        torch.Tensor: ``(B, T, C)``.
    """
    def __init__(self, embed_dim, num_heads, dropout=0.0, mask_type=None, qkv_bias=False,
                 device='cpu', dtype=torch.float32, backend: Literal["sdpa", "flex"] = "sdpa",
                 combine: Literal["or", "and"] = "or", **mask_kwargs):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        if backend not in ("sdpa", "flex"):
            raise ValueError(f"backend must be 'sdpa' or 'flex', got {backend!r}")

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.mask_type = mask_type
        self.combine = combine
        self.backend = backend
        self.device = device
        self.dtype = dtype
        self.mask_kwargs = mask_kwargs

        # Projection layers
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=qkv_bias, device=device, dtype=self.dtype)
        self.kv_proj = nn.Linear(embed_dim, self.head_dim * 2, bias=qkv_bias, device=device, dtype=self.dtype)
        
        # Output final projection
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=qkv_bias, device=device, dtype=self.dtype)
        
        # Dropout applied to the attention weights (sdpa backend only)
        self.dropout_p = dropout
        
        self._causal_mask_cache = OrderedDict()

    def _get_or_create_mask(self, seq_len: int, device):
        if self.backend == "sdpa":
            return _get_attention_mask(
                self._causal_mask_cache, self.mask_type, seq_len, device,
                combine=self.combine, **self.mask_kwargs,
            )
        return _get_block_mask(
            self._causal_mask_cache, self.mask_type, seq_len, device,
            combine=self.combine, **self.mask_kwargs,
        )
    
    def _precompute_theta_position_frequency(self, head_dim: int, seq_len: int, device: torch.device,theta: float = 10000.0):
        return _build_rope_frequency(head_dim, seq_len, device, self.dtype, theta=theta)
    
    def forward(self, x, mask=True, theta: float = 10000.0):
        B, T, C = x.shape

        # Project
        q = self.q_proj(x)                    # (B, T, C)
        kv = self.kv_proj(x)                  # (B, T, 2*D)
        k, v = kv.split(self.head_dim, dim=-1)  # each (B, T, D)

        # Reshape
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, T, D)

        # Single KV head -- left un-expanded (B, 1, T, D)
        k = k.unsqueeze(1)  # (B, 1, T, D)
        v = v.unsqueeze(1)  # (B, 1, T, D)

        # RoPE
        freq = self._precompute_theta_position_frequency(
            self.head_dim,
            T,
            device=x.device,
            theta=theta,
        )

        q = _apply_rotary_position_embedding(q, freq)
        k = _apply_rotary_position_embedding(k, freq)

        # No .expand() -- enable_gqa broadcasts K/V from 1 head to H heads
        # inside the kernel.
        attn_mask = self._get_or_create_mask(seq_len=T, device=x.device) if mask else None

        if self.backend == "sdpa":
            out = _run_sdpa(q, k, v, attn_mask=attn_mask, dropout_p=self.dropout_p, enable_gqa=True)
        else:
            out = _run_flex_attention(q, k, v, block_mask=attn_mask, enable_gqa=True)

        # Merge heads
        out = out.transpose(1, 2).reshape(B, T, C)

        return self.out_proj(out)

class Group_query_Attention(nn.Module):
    """Grouped-Query Attention (GQA): intermediate between MHA and MQA.

    Backend note:
        K/V stay at ``num_kv_heads`` heads -- no ``repeat_interleave`` up
        front. Both SDPA (``enable_gqa=True``, torch >= 2.5) and
        FlexAttention (``enable_gqa=True``) broadcast K/V to the query-head
        groups internally, which is faster and uses less memory than
        materializing the repeated tensors first, and scales better as
        ``num_queries_per_kv`` grows.

    Constructor args:
        embed_dim (int, required).
        num_query_heads (int, required): Rule ``embed_dim % num_query_heads == 0``.
        num_kv_heads (int, required): Rule ``num_query_heads % num_kv_heads == 0``.
        qkv_bias (bool, optional, default=False).
        dropout (float, optional, default=0.0): SDPA-only.
        device (optional, default='cpu').
        dtype (optional, default=torch.float32).
        backend (Literal["sdpa", "flex"], optional, default="sdpa").
        combine (Literal["or", "and"], optional, default="or").
        mask_type ([str], optional, default=['causal']): 'causal' or 'sliding_window'.

    Forward args:
        x (torch.Tensor): ``(B, T, C)``.
        mask (bool, optional, default=True).

    Returns:
        torch.Tensor: ``(B, T, C)``.

    Example:
        >>> gqa = Group_query_Attention(embed_dim=512, num_query_heads=8, num_kv_heads=2)
        >>> y = gqa(torch.randn(2, 128, 512))
        >>> gqa_flex = Group_query_Attention(embed_dim=512, num_query_heads=8, num_kv_heads=2,
        ...                                   backend="flex", mask_type="sliding_window", window_size=64)
    """
    def __init__(self, embed_dim, num_query_heads, num_kv_heads, qkv_bias=False, dropout=0.0,
                 mask_type=None, device='cpu', dtype=torch.float32,
                 backend: Literal["sdpa", "flex"] = "sdpa", combine: Literal["or", "and"] = "or",
                 **mask_kwargs):
        super().__init__()
        assert embed_dim % num_query_heads == 0, "embed_dim must be divisible by num_query_heads"
        assert num_query_heads % num_kv_heads == 0, "num_query_heads must be divisible by num_kv_heads"
        if backend not in ("sdpa", "flex"):
            raise ValueError(f"backend must be 'sdpa' or 'flex', got {backend!r}")

        self.dtype = dtype
        self.device = device
        self.embed_dim = embed_dim
        self.num_query_heads = num_query_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = embed_dim // num_query_heads
        self.num_queries_per_kv = num_query_heads // num_kv_heads
        self.mask_type = mask_type
        self.combine = combine
        self.backend = backend
        self.mask_kwargs = mask_kwargs

        # Projection layers
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=qkv_bias, device=device, dtype=dtype)
        self.kv_proj = nn.Linear(embed_dim, num_kv_heads * self.head_dim * 2, bias=qkv_bias, device=device, dtype=dtype)
        
        # Output final projection
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=qkv_bias, device=device, dtype=dtype)
        
        # Dropout applied to the attention weights (sdpa backend only)
        self.dropout_p = dropout
        
        self._causal_mask_cache = OrderedDict()

    def _get_or_create_mask(self, seq_len: int, device):
        if self.backend == "sdpa":
            return _get_attention_mask(
                self._causal_mask_cache, self.mask_type, seq_len, device,
                combine=self.combine, **self.mask_kwargs,
            )
        return _get_block_mask(
            self._causal_mask_cache, self.mask_type, seq_len, device,
            combine=self.combine, **self.mask_kwargs,
        )

    def forward(self, x, mask=True):
        B, T, C = x.shape

        # Project Q, K, V
        q = self.q_proj(x)  # (B, T, C)
        kv = self.kv_proj(x)  # (B, T, 2 * num_kv_heads * head_dim)
        kv_dim = self.num_kv_heads * self.head_dim
        k, v = kv.split(kv_dim, dim=-1)
        
        # Reshape projections
        q = q.view(B, T, self.num_query_heads, self.head_dim).transpose(1, 2)  # (B, num_query_heads, T, head_dim)
        k = k.view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)     # (B, num_kv_heads, T, head_dim)
        v = v.view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)     # (B, num_kv_heads, T, head_dim)

        # No repeat_interleave here -- enable_gqa lets the kernel broadcast
        # num_kv_heads -> num_query_heads internally instead of materializing
        # the repeated K/V.

        attn_mask = self._get_or_create_mask(seq_len=T, device=x.device) if mask else None

        if self.backend == "sdpa":
            out = _run_sdpa(q, k, v, attn_mask=attn_mask, dropout_p=self.dropout_p, enable_gqa=True)
        else:
            out = _run_flex_attention(q, k, v, block_mask=attn_mask, enable_gqa=True)

        out = out.transpose(1, 2).contiguous().view(B, T, C)

        return self.out_proj(out)
    
class Group_query_Attention_With_RoPE(nn.Module):
    """GQA with RoPE for relative-position-aware grouped attention.

    Backend note:
        Same treatment as ``Group_query_Attention``: K/V stay at
        ``num_kv_heads``, RoPE is applied before any head-count change (there
        isn't one anymore), and ``enable_gqa=True`` does the broadcast to
        ``num_query_heads`` inside the kernel for both backends.

    Constructor args:
        embed_dim (int, required).
        num_query_heads (int, required): ``embed_dim % num_query_heads == 0``.
        num_kv_heads (int, required): ``num_query_heads % num_kv_heads == 0``.
        qkv_bias (bool, optional, default=False).
        dropout (float, optional, default=0.0): SDPA-only.
        device (optional, default='cpu').
        dtype (optional, default=torch.float32).
        backend (Literal["sdpa", "flex"], optional, default="sdpa").
        combine (Literal["or", "and"], optional, default="or").
        mask_type ([str], optional, default=['causal']): 'causal' or 'sliding_window'.

    Rules:
        RoPE requires even head dimension.

    Forward args:
        x (torch.Tensor): ``(B, T, C)``.
        mask (bool, optional, default=True).

    Returns:
        torch.Tensor: ``(B, T, C)``.
    """
    def __init__(self, embed_dim, num_query_heads, num_kv_heads, qkv_bias=False, dropout=0.0,
                 mask_type=None, device='cpu', dtype=torch.float32,
                 backend: Literal["sdpa", "flex"] = "sdpa", combine: Literal["or", "and"] = "or",
                 **mask_kwargs):
        super().__init__()
        assert embed_dim % num_query_heads == 0, "embed_dim must be divisible by num_query_heads"
        assert num_query_heads % num_kv_heads == 0, "num_query_heads must be divisible by num_kv_heads"
        if backend not in ("sdpa", "flex"):
            raise ValueError(f"backend must be 'sdpa' or 'flex', got {backend!r}")

        self.dtype = dtype
        self.device = device
        self.embed_dim = embed_dim
        self.num_query_heads = num_query_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = embed_dim // num_query_heads
        self.num_queries_per_kv = num_query_heads // num_kv_heads
        self.mask_type = mask_type
        self.combine = combine
        self.backend = backend
        self.mask_kwargs = mask_kwargs

        # Projection layers
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=qkv_bias, device=device, dtype=dtype)
        self.kv_proj = nn.Linear(embed_dim, num_kv_heads * self.head_dim * 2, bias=qkv_bias, device=device, dtype=dtype)
        
        # Output final projection
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=qkv_bias, device=device, dtype=dtype)
        
        # Dropout applied to the attention weights (sdpa backend only)
        self.dropout_p = dropout

        self._causal_mask_cache = OrderedDict()

    def _get_or_create_mask(self, seq_len: int, device):
        if self.backend == "sdpa":
            return _get_attention_mask(
                self._causal_mask_cache, self.mask_type, seq_len, device,
                combine=self.combine, **self.mask_kwargs,
            )
        return _get_block_mask(
            self._causal_mask_cache, self.mask_type, seq_len, device,
            combine=self.combine, **self.mask_kwargs,
        )
    
    def _precompute_theta_position_frequency(self, head_dim: int, seq_len: int, device: torch.device,theta: float = 10000.0):
        return _build_rope_frequency(head_dim, seq_len, device, self.dtype, theta=theta)
    
    def forward(self, x, mask=True, theta: float=10000.0):
        B, T, C = x.shape

        # Project Q, K, V
        q = self.q_proj(x)  # (B, T, C)
        kv = self.kv_proj(x)  # (B, T, 2 * num_kv_heads * head_dim)
        kv_dim = self.num_kv_heads * self.head_dim
        k, v = kv.split(kv_dim, dim=-1)
        
        # Reshape projections
        q = q.view(B, T, self.num_query_heads, self.head_dim).transpose(1, 2)  # (B, num_query_heads, T, head_dim)
        k = k.view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)     # (B, num_kv_heads, T, head_dim)
        v = v.view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)     # (B, num_kv_heads, T, head_dim)

        freq = self._precompute_theta_position_frequency(self.head_dim, T, device=x.device,theta=theta)
        q = _apply_rotary_position_embedding(q, freq)
        k = _apply_rotary_position_embedding(k, freq)
        
        # No repeat_interleave -- enable_gqa broadcasts num_kv_heads ->
        # num_query_heads internally.

        attn_mask = self._get_or_create_mask(seq_len=T, device=x.device) if mask else None

        if self.backend == "sdpa":
            out = _run_sdpa(q, k, v, attn_mask=attn_mask, dropout_p=self.dropout_p, enable_gqa=True)
        else:
            out = _run_flex_attention(q, k, v, block_mask=attn_mask, enable_gqa=True)
        
        out = out.transpose(1, 2).contiguous().view(B, T, C)

        return self.out_proj(out)

class kv_cache_multihead(nn.Module):
    """MHA with persistent KV cache for incremental decoding.

    Constructor args:
        embed_dim (int, required).
        num_heads (int, required): ``embed_dim % num_heads == 0``.
        batch_size (int, required): Preallocated cache batch capacity.
        kv_seq_len (int, required): Maximum cache sequence length reference.
        qkv_bias (bool, optional, default=False).
        dropout (float, optional, default=0.0).
        device (optional, default='cpu').
        dtype (optional, default=torch.float32).
        cache_policy (KVCachePolicy | None, optional): defaults to StaticKVCache.

    Forward args:
        x (torch.Tensor): ``(B, T, C)`` token chunk.
        start_pos (int, required): Write offset in cache.
        mask (bool, optional, default=True): Causal masking over cache span.
        rope (bool, optional, default=True): Enable RoPE before cache write.

    Returns:
        torch.Tensor: ``(B, T, C)``.
    """
    def __init__(self, embed_dim, num_heads, batch_size, kv_seq_len, mask_type=None,
                 qkv_bias=False, dropout=0.0, device='cpu', dtype=torch.float32,
                 cache_policy=None, **mask_kwargs):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.device = device
        self.dtype = dtype
        self.mask_type = mask_type
        self.mask_kwargs = mask_kwargs

        #  QKV projection
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=qkv_bias, device=device, dtype=dtype)
        self.kv_proj = nn.Linear(embed_dim, embed_dim * 2, bias=qkv_bias, device=device, dtype=dtype)

        # Final output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=qkv_bias, device=device, dtype=dtype)

        # Dropout applied to the attention weights
        self.dropout_p = dropout

        self.kv_seq_len = kv_seq_len
        self._causal_mask_cache = OrderedDict()

        # Cache storage/retrieval is delegated entirely to a KVCachePolicy —
        # this module no longer owns cache_keys/cache_values directly.
        self.cache = cache_policy or StaticKVCache()
        self.cache.allocate(batch_size=batch_size, num_heads=num_heads,
                             head_dim=self.head_dim, max_len=kv_seq_len,
                             device=device, dtype=dtype)

    def _precompute_theta_position_frequency(self, head_dim: int, seq_len: int, device: torch.device, theta: float = 10000.0):
        return _build_rope_frequency(head_dim, seq_len, device, self.dtype, theta=theta)

    def _get_or_create_kv_mask(self, T, S, start_pos, device, max_cache_size=64):
        key = (T, S, start_pos, str(device))

        if key in self._causal_mask_cache:
            self._causal_mask_cache.move_to_end(key)  # mark as recently used
            return self._causal_mask_cache[key]

        if len(self._causal_mask_cache) >= max_cache_size:
            self._causal_mask_cache.popitem(last=False)  # evict least-recently-used

        i = torch.arange(T, device=device).unsqueeze(1)
        j = torch.arange(S, device=device).unsqueeze(0)
        visible = j <= (start_pos + i)
        self._causal_mask_cache[key] = visible

        return self._causal_mask_cache[key]

    def reset_cache(self):
        self.cache.reset()
        self._causal_mask_cache.clear()

    def forward(self, x: torch.Tensor, start_pos: int = 0, mask: bool = True, rope: bool = True, theta: float = 10000.0):
        B, T, C = x.shape

        assert C == self.embed_dim, "Input embed_dim mismatch"
        end_pos = start_pos + T
        assert end_pos <= self.kv_seq_len, (
            f"KV cache capacity exceeded: start_pos={start_pos} + T={T} = {end_pos} "
            f"> cache length {self.kv_seq_len}"
        )

        # Project Q, K, V
        q = self.q_proj(x)  # (B, T, C)
        kv = self.kv_proj(x)  # (B, T, 2 * C)
        k, v = kv.chunk(2, dim=-1)

        # Reshape to multi-head
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2) # (B, H, T, D)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        # Rotary Position Embedding (correct for KV cache)
        if rope:
            freq = self._precompute_theta_position_frequency(self.head_dim, end_pos, device=x.device, theta=theta)
            q = _apply_rotary_position_embedding(q, freq[start_pos:end_pos])
            k = _apply_rotary_position_embedding(k, freq[start_pos:end_pos])

        # Cache owns detach/dtype-guard/grad-safety entirely — pass raw k, v
        self.cache.update(k, v, start_pos)
        k_full, v_full = self.cache.get_kv(start_pos, end_pos)
        # (no further grad handling needed here — get_kv() already spliced it)

        # Rectangular causal mask (T, S)
        attn_mask = None
        if mask:
            attn_mask = self._get_or_create_kv_mask(T, end_pos, start_pos, device=x.device)

        # Scaled Dot Product Attention (Flash / MemEff)
        context = F.scaled_dot_product_attention(
            q,
            k_full,
            v_full,
            attn_mask=attn_mask,   # (T, S)
            dropout_p=self.dropout_p
        )

        # Merge heads + output projection
        out = context.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(out)

class kv_cache_group_query(nn.Module):  
    """GQA with KV cache for production-grade decoding throughput.

    Backend note:
        This class is SDPA-only. Incremental decoding grows the cache's
        span ``S = start_pos + T`` by ``T`` on every call, and a
        FlexAttention ``BlockMask`` is compiled for a fixed ``(Q_LEN, KV_LEN)``
        pair -- rebuilding/recompiling one every decode step would cost far
        more than it saves, so this class keeps ``F.scaled_dot_product_attention``
        directly. It still gets the ``enable_gqa`` speedup: K/V are cached and
        returned at their native ``num_kv_heads`` head count and are no
        longer expanded to ``num_query_heads`` via ``repeat_interleave``
        before the attention call.

    Constructor args:
        embed_dim (int, required).
        num_query_heads (int, required): ``embed_dim % num_query_heads == 0``.
        num_kv_heads (int, required): ``num_query_heads % num_kv_heads == 0``.
        kv_seq_len (int, required): Maximum cache length reference.
        batch_size (int, required): Cache batch capacity.
        qkv_bias (bool, optional, default=False).
        dropout (float, optional, default=0.0).
        device (optional, default='cpu').
        dtype (optional, default=torch.float32).

    Forward args:
        x (torch.Tensor): ``(B, T, C)``.
        start_pos (int, required): Cache write position.
        mask (bool, optional, default=True).
        rope (bool, optional, default=True).

    Returns:
        torch.Tensor: ``(B, T, C)``.
    """
    def __init__(self, embed_dim, num_query_heads, num_kv_heads, kv_seq_len, batch_size, mask_type=None, qkv_bias=False, dropout=0.0, device='cpu', dtype=torch.float32, **mask_kwargs):
        super().__init__()
        assert embed_dim % num_query_heads == 0, "embed_dim must be divisible by num_query_heads"
        assert num_query_heads % num_kv_heads == 0, "num_query_heads must be divisible by num_kv_heads"

        self.embed_dim = embed_dim
        self.num_query_heads = num_query_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = embed_dim // num_query_heads
        self.num_queries_per_kv = num_query_heads // num_kv_heads
        self.kv_seq_len = kv_seq_len
        self.device = device
        self.dtype = dtype
        self.mask_type = mask_type
        self.mask_kwargs = mask_kwargs

        # Linear projections: Q from full dim, KV from reduced dim
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=qkv_bias, device=device, dtype=dtype)
        self.kv_proj = nn.Linear(embed_dim, num_kv_heads * self.head_dim * 2, bias=qkv_bias, device=device, dtype=dtype)
        
        # Output final projection        
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=qkv_bias, device=device, dtype=dtype)
        
        # Dropout applied to the attention weights
        self.dropout_p = dropout
        
        self._causal_mask_cache = OrderedDict()

        self.cache = StaticKVCache()
        self.cache.allocate(batch_size=batch_size, num_heads=num_kv_heads,
                             head_dim=self.head_dim, max_len=kv_seq_len,
                             device=device, dtype=dtype)

    def _precompute_theta_position_frequency(self, head_dim: int, seq_len: int, device: torch.device, theta: float = 10000.0):
        return _build_rope_frequency(head_dim, seq_len, device, self.dtype, theta=theta)

    def _get_or_create_kv_mask(self, T, S, start_pos, device, max_cache_size=64):
        key = (T, S, start_pos, str(device))

        if key in self._causal_mask_cache:
            self._causal_mask_cache.move_to_end(key)  # mark as recently used
            return self._causal_mask_cache[key]

        if len(self._causal_mask_cache) >= max_cache_size:
            self._causal_mask_cache.popitem(last=False)  # evict least-recently-used

        i = torch.arange(T, device=device).unsqueeze(1)
        j = torch.arange(S, device=device).unsqueeze(0)
        visible = j <= (start_pos + i)
        self._causal_mask_cache[key] = visible

        return self._causal_mask_cache[key]
    
    def reset_cache(self):
        self.cache.reset()
        self._causal_mask_cache.clear()

    def forward(self, x, start_pos=0, mask=True, rope=True, theta: float = 10000.0):
        B, T, C = x.shape
        assert C == self.embed_dim, "Input embed_dim mismatch"
        end_pos = start_pos + T
        assert end_pos <= self.kv_seq_len, (
            f"KV cache capacity exceeded: start_pos={start_pos} + T={T} = {end_pos} "
            f"> cache length {self.kv_seq_len}"
        )

        # Project Q, K, V
        q = self.q_proj(x)
        kv = self.kv_proj(x)
        k, v = kv.chunk(2, dim=-1)  # each (B, T, num_kv_heads * head_dim)

        q = q.view(B, T, self.num_query_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # RoPE (correct for KV cache)
        if rope:
            freq = self._precompute_theta_position_frequency(self.head_dim, end_pos, device=x.device, theta=theta)
            q = _apply_rotary_position_embedding(q, freq[start_pos:end_pos])
            k = _apply_rotary_position_embedding(k, freq[start_pos:end_pos])

        # Cache owns the detach/grad-safety decision entirely now — pass raw k, v
        self.cache.update(k, v, start_pos)
        k_full, v_full = self.cache.get_kv(start_pos, end_pos)   # <-- start_pos, not 0
        # (no further grad handling needed here — get_kv() already spliced it)

        # k_full/v_full stay at num_kv_heads heads -- NOT expanded to
        # num_query_heads. enable_gqa=True below lets SDPA broadcast the
        # head groups internally instead of paying for a repeat_interleave
        # on every single decode step.

        attn_mask = None
        if mask:
            attn_mask = self._get_or_create_kv_mask(T, end_pos, start_pos, device=x.device)

        context = _run_sdpa(q, k_full, v_full, attn_mask=attn_mask, dropout_p=self.dropout_p, enable_gqa=True)

        out = context.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(out)


# New canonical class names (Step 4 rename)
SelfAttention = Self_Attention
MultiHeadAttention = Multi_Head_Attention
MultiHeadAttentionWithRoPE = Multi_Head_Attention_With_RoPE
CrossMultiHeadAttention = Cross_MultiHead_Attention
MultiQueryAttention = Multi_query_Attention
MultiQueryAttentionWithRoPE = Multi_query_Attention_With_RoPE
GroupQueryAttention = Group_query_Attention
GroupQueryAttentionWithRoPE = Group_query_Attention_With_RoPE
KVCacheMultiHead = kv_cache_multihead
KVCacheGroupQuery = kv_cache_group_query