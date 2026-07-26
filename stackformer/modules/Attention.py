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
- ``D``: head dimension, usually ``C // H``

Implementation notes:
- Inputs are moved to ``device``/``dtype`` configured in each module.
- Causal masks are cached by sequence shape to reduce allocation overhead.
- RoPE modules require even head dimension (enforced with assertions).

Quick start:
    >>> import torch
    >>> from stackformer.modules.Attention import Multi_Head_Attention
    >>> x = torch.randn(2, 32, 256)
    >>> attn = Multi_Head_Attention(embed_dim=256, num_heads=8)
    >>> y = attn(x, mask=True)
    >>> y.shape
    torch.Size([2, 32, 256])
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from stackformer.utils.attn_utils import _run_sdpa, _get_attention_mask 


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
    def __init__(self, embed_dim, num_heads, dropout=0.0, mask_type=None,qkv_bias=False,device='cpu', dtype=torch.float32, **mask_kwargs):
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
    """Multi-Query Attention (MQA): many query heads, shared K/V head.

    Constructor args:
        embed_dim (int, required).
        num_heads (int, required): Number of query heads. Rule:
            ``embed_dim % num_heads == 0``.
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
        self.head_dim = embed_dim // num_heads
        self.mask_type = mask_type
        self.device = device
        self.dtype = dtype
        self.mask_kwargs = mask_kwargs

        # Projection layers
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=qkv_bias, device=device, dtype=self.dtype)
        self.kv_proj = nn.Linear(embed_dim, self.head_dim * 2, bias=qkv_bias, device=device, dtype=self.dtype)
        
        # Output final projection
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=qkv_bias, device=device, dtype=self.dtype)
        
        # Dropout applied to the attention weights
        self.dropout_p = dropout
        
        # Cache for causal masks (for autoregressive models)
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

        # Project
        q = self.q_proj(x)                    # (B, T, C)
        kv = self.kv_proj(x)                  # (B, T, 2*D)
        k, v = kv.split(self.head_dim, dim=-1) # (B, T, D) each

        # Multi-head queries
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2) # (B, H, T, D)
        # Shared K/V
        k = k.unsqueeze(1)                    # (B, 1, T, D)
        v = v.unsqueeze(1)                    # (B, 1, T, D)

        # Broadcast to all query heads (no memory allocation)
        k = k.expand(-1, self.num_heads, -1, -1)
        v = v.expand(-1, self.num_heads, -1, -1)

        attn_mask = None
        if mask:
            attn_mask = self._get_or_create_mask(
                seq_len=T,
                device=x.device,
            )

        out = _run_sdpa(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=self.dropout_p,
        )

        out = out.transpose(1, 2).reshape(B, T, C)

        return self.out_proj(out)
    
class Multi_query_Attention_With_RoPE(nn.Module):
    """MQA with RoPE on queries and shared keys.

    Constructor args:
        embed_dim (int, required).
        num_heads (int, required): ``embed_dim % num_heads == 0`` and even
            ``head_dim`` for RoPE.
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
        self.head_dim = embed_dim // num_heads
        self.mask_type = mask_type
        self.device = device
        self.dtype = dtype
        self.mask_kwargs = mask_kwargs

        # Projection layers
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=qkv_bias, device=device, dtype=self.dtype)
        self.kv_proj = nn.Linear(embed_dim, self.head_dim * 2, bias=qkv_bias, device=device, dtype=self.dtype)
        
        # Output final projection
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=qkv_bias, device=device, dtype=self.dtype)
        
        # Dropout applied to the attention weights
        self.dropout_p = dropout
        
        # Cache for causal masks (for autoregressive models)
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
    
    def forward(self, x, mask=True, theta: float = 10000.0):
        B, T, C = x.shape

        # Project
        q = self.q_proj(x)                    # (B, T, C)
        kv = self.kv_proj(x)                  # (B, T, 2*D)
        k, v = kv.split(self.head_dim, dim=-1)  # each (B, T, D)

        # Reshape
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, T, D)

        # Single KV head
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

        # Broadcast KV to all query heads (no memory copy)
        k = k.expand(-1, self.num_heads, -1, -1)
        v = v.expand(-1, self.num_heads, -1, -1)

        # Causal mask
        attn_mask = None
        if mask:
            attn_mask = self._get_or_create_mask(
                seq_len=T,
                device=x.device,
            )

        # SDPA
        out = _run_sdpa(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=self.dropout_p,
        )

        # Merge heads
        out = out.transpose(1, 2).reshape(B, T, C)

        return self.out_proj(out)

class Group_query_Attention(nn.Module):
    """Grouped-Query Attention (GQA): intermediate between MHA and MQA.

    Constructor args:
        embed_dim (int, required).
        num_query_heads (int, required): Rule ``embed_dim % num_query_heads == 0``.
        num_kv_heads (int, required): Rule ``num_query_heads % num_kv_heads == 0``.
        qkv_bias (bool, optional, default=False).
        dropout (float, optional, default=0.0).
        device (optional, default='cpu').
        dtype (optional, default=torch.float32).
        mask_type ([str], optional, default=['causal']): 'causal' or 'sliding_window'.

    Forward args:
        x (torch.Tensor): ``(B, T, C)``.
        mask (bool, optional, default=True).

    Returns:
        torch.Tensor: ``(B, T, C)``.
    """
    def __init__(self, embed_dim, num_query_heads, num_kv_heads, qkv_bias=False, dropout=0.0, mask_type=None, device='cpu', dtype=torch.float32, **mask_kwargs):
        super().__init__()
        assert embed_dim % num_query_heads == 0, "embed_dim must be divisible by num_query_heads"
        assert num_query_heads % num_kv_heads == 0, "num_query_heads must be divisible by num_kv_heads"

        self.dtype = dtype
        self.device = device
        self.embed_dim = embed_dim
        self.num_query_heads = num_query_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = embed_dim // num_query_heads
        self.num_queries_per_kv = num_query_heads // num_kv_heads
        self.mask_type = mask_type
        self.mask_kwargs = mask_kwargs

        # Projection layers
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=qkv_bias, device=device, dtype=dtype)
        self.kv_proj = nn.Linear(embed_dim, num_kv_heads * self.head_dim * 2, bias=qkv_bias, device=device, dtype=dtype)
        
        # Output final projection
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=qkv_bias, device=device, dtype=dtype)
        
        # Dropout applied to the attention weights
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
        q = self.q_proj(x)  # (B, T, C)
        kv = self.kv_proj(x)  # (B, T, 2 * num_kv_heads * head_dim)
        kv_dim = self.num_kv_heads * self.head_dim
        k, v = kv.split(kv_dim, dim=-1)
        
        # Reshape projections
        q = q.view(B, T, self.num_query_heads, self.head_dim).transpose(1, 2)  # (B, num_query_heads, T, head_dim)
        k = k.view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)     # (B, num_kv_heads, T, head_dim)
        v = v.view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # Repeat keys and values for each query head group
        k = k.repeat_interleave(self.num_queries_per_kv, dim=1)  # (B, num_query_heads, T, head_dim)
        v = v.repeat_interleave(self.num_queries_per_kv, dim=1)

        causal_mask = None
        # Apply causal mask if needed
        if mask:
            causal_mask = self._get_or_create_mask(seq_len=T, device=x.device)  # (T, T)

        out = _run_sdpa(
            q, k, v,
            attn_mask = causal_mask,
            dropout_p = self.dropout_p
        )

        out = out.transpose(1, 2).contiguous().view(B, T, C)

        return self.out_proj(out)
    
class Group_query_Attention_With_RoPE(nn.Module):
    """GQA with RoPE for relative-position-aware grouped attention.

    Constructor args:
        embed_dim (int, required).
        num_query_heads (int, required): ``embed_dim % num_query_heads == 0``.
        num_kv_heads (int, required): ``num_query_heads % num_kv_heads == 0``.
        qkv_bias (bool, optional, default=False).
        dropout (float, optional, default=0.0).
        device (optional, default='cpu').
        dtype (optional, default=torch.float32).
        mask_type ([str], optional, default=['causal']): 'causal' or 'sliding_window'.

    Rules:
        RoPE requires even head dimension.

    Forward args:
        x (torch.Tensor): ``(B, T, C)``.
        mask (bool, optional, default=True).

    Returns:
        torch.Tensor: ``(B, T, C)``.
    """
    def __init__(self, embed_dim, num_query_heads, num_kv_heads, qkv_bias=False, dropout=0.0, mask_type=None, device='cpu', dtype=torch.float32, **mask_kwargs):
        super().__init__()
        assert embed_dim % num_query_heads == 0, "embed_dim must be divisible by num_query_heads"
        assert num_query_heads % num_kv_heads == 0, "num_query_heads must be divisible by num_kv_heads"

        self.dtype = dtype
        self.device = device
        self.embed_dim = embed_dim
        self.num_query_heads = num_query_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = embed_dim // num_query_heads
        self.num_queries_per_kv = num_query_heads // num_kv_heads
        self.mask_type = mask_type
        self.mask_kwargs = mask_kwargs

        # Projection layers
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=qkv_bias, device=device, dtype=dtype)
        self.kv_proj = nn.Linear(embed_dim, num_kv_heads * self.head_dim * 2, bias=qkv_bias, device=device, dtype=dtype)
        
        # Output final projection
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=qkv_bias, device=device, dtype=dtype)
        
        # Dropout applied to the attention weights
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
        v = v.view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)

        freq = self._precompute_theta_position_frequency(self.head_dim, T, device=x.device,theta=theta)
        q = _apply_rotary_position_embedding(q, freq)
        k = _apply_rotary_position_embedding(k, freq)
        
        # Repeat keys and values for each query head group
        k = k.repeat_interleave(self.num_queries_per_kv, dim=1)  # (B, num_query_heads, T, head_dim)
        v = v.repeat_interleave(self.num_queries_per_kv, dim=1)
        
        
        causal_mask = None
        # Apply causal mask if needed
        if mask:
            causal_mask = self._get_or_create_mask(seq_len=T, device=x.device)  # (T, T)

        out = _run_sdpa(
            q, k, v,
            attn_mask = causal_mask,
            dropout_p = self.dropout_p
        )
        
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

    Forward args:
        x (torch.Tensor): ``(B, T, C)`` token chunk.
        start_pos (int, required): Write offset in cache.
        mask (bool, optional, default=True): Causal masking over cache span.
        rope (bool, optional, default=True): Enable RoPE before cache write.

    Returns:
        torch.Tensor: ``(B, T, C)``.
    """
    def __init__(self, embed_dim, num_heads, batch_size, kv_seq_len, mask_type=None, qkv_bias=False, dropout=0.0, device='cpu', dtype=torch.float32, **mask_kwargs):
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

        # KV Cache registered as buffers so they move with .to(device) and are
        # included in state_dict. persistent=False keeps them out of checkpoints
        # (they are transient inference state, not learned parameters).
        self.register_buffer(
        "cache_keys",
        torch.empty(batch_size,num_heads,kv_seq_len,self.head_dim,device=self.device,dtype=self.dtype,),
        persistent=False,
        )

        self.register_buffer(
            "cache_values",
            torch.empty(batch_size,num_heads,kv_seq_len,self.head_dim,device=self.device,dtype=self.dtype,),
            persistent=False,
        )
        self.kv_seq_len = kv_seq_len
        self._causal_mask_cache = OrderedDict()

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
        self.cache_keys.zero_()
        self.cache_values.zero_()
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
            # apply only the slice corresponding to current tokens
            q = _apply_rotary_position_embedding(q, freq[start_pos:end_pos])
            k = _apply_rotary_position_embedding(k, freq[start_pos:end_pos])

        # Write KV cache — detach only at inference to avoid blocking gradients
        # during training (torch.is_grad_enabled() is False inside torch.no_grad())
        _k_to_store = k.detach() if not torch.is_grad_enabled() else k
        _v_to_store = v.detach() if not torch.is_grad_enabled() else v

        # Guard against silent dtype mismatch under autocast/mixed precision
        assert _k_to_store.dtype == self.cache_keys.dtype, (
            f"KV cache dtype mismatch: got {_k_to_store.dtype}, "
            f"cache is {self.cache_keys.dtype}. Cast inputs or recreate the cache "
            f"with the desired dtype."
        )

        self.cache_keys[:B, :, start_pos:end_pos] = _k_to_store
        self.cache_values[:B, :, start_pos:end_pos] = _v_to_store
        
        # Read full cached KV
        k_full = self.cache_keys[:B, :, :end_pos]
        v_full = self.cache_values[:B, :, :end_pos]

        # RATIONALE (Gradient Flow & Autograd Safety):
        # During training (or when gradients are enabled), writing in-place to pre-allocated buffers
        # creates dependencies across time steps that would corrupt PyTorch's autograd graph.
        # To ensure correct backpropagation through current-step keys/values while reusing
        # memory, we detach and clone the historical prefix tokens (0 to start_pos), then splice in
        # the live autograd computation nodes for the current chunk (start_pos to end_pos).
        if torch.is_grad_enabled():
            k_full = k_full.detach().clone()
            v_full = v_full.detach().clone()
            k_full[:, :, start_pos:end_pos] = _k_to_store
            v_full[:, :, start_pos:end_pos] = _v_to_store

        # Rectangular causal mask (T, S)
        attn_mask = None
        if mask:
            attn_mask = self._get_or_create_kv_mask(T, end_pos, start_pos, device=x.device)


        # Scaled Dot Product Attention (Flash / MemEff)        
        context = F.scaled_dot_product_attention(
            q,
            k_full,
            v_full,
            attn_mask = attn_mask,   # (T, S)
            dropout_p = self.dropout_p
        )

        # Merge heads + output projection        
        out = context.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(out)

class kv_cache_group_query(nn.Module):  
    """GQA with KV cache for production-grade decoding throughput.

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

        # KV Cache registered as buffers so they move with .to(device) and are
        # included in state_dict. persistent=False keeps them out of checkpoints.
        self.register_buffer(
            "cache_keys",
            torch.empty(batch_size, num_kv_heads, kv_seq_len, self.head_dim, device=self.device, dtype=self.dtype),
            persistent=False,
        )
        self.register_buffer(
            "cache_values",
            torch.empty(batch_size, num_kv_heads, kv_seq_len, self.head_dim, device=self.device, dtype=self.dtype),
            persistent=False,
        )

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
        self.cache_keys.zero_()
        self.cache_values.zero_()
        self._causal_mask_cache.clear()

    def forward(self, x, start_pos=0, mask=True, rope=True, theta: float = 10000.0):
        B, T, C = x.shape
        end_pos = start_pos + T
        assert end_pos <= self.kv_seq_len, (
            f"KV cache capacity exceeded: start_pos={start_pos} + T={T} = {end_pos} "
            f"> cache length {self.kv_seq_len}"
        )

        # Project Q, K, V
        q = self.q_proj(x)
        kv = self.kv_proj(x)
        k, v = kv.chunk(2, dim=-1)  # each (B, T, num_kv_heads * head_dim)
        # Reshape
        # Q: (B, T, QH, D)  K/V: (B, T, KVH, D)
        q = q.view(B, T, self.num_query_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        
        # RoPE (correct for KV cache)
        if rope:
            freq = self._precompute_theta_position_frequency(self.head_dim,end_pos,device=x.device,theta=theta,)
            q = _apply_rotary_position_embedding(q,freq[start_pos:end_pos],)
            k = _apply_rotary_position_embedding(k,freq[start_pos:end_pos],)

        # Write KV cache — detach only at inference to avoid blocking gradients
        _k_to_store = k.detach() if not torch.is_grad_enabled() else k
        _v_to_store = v.detach() if not torch.is_grad_enabled() else v
        self.cache_keys[:B, :, start_pos:end_pos] = _k_to_store
        self.cache_values[:B, :, start_pos:end_pos] = _v_to_store

        # Read full KV
        k_full = self.cache_keys[:B, :, :end_pos]
        v_full = self.cache_values[:B, :, :end_pos]

        # RATIONALE (Gradient Flow & Autograd Safety):
        # During training (or when gradients are enabled), writing in-place to pre-allocated buffers
        # creates dependencies across time steps that would corrupt PyTorch's autograd graph.
        # To ensure correct backpropagation through current-step keys/values while reusing
        # memory, we detach and clone the historical prefix tokens (0 to start_pos), then splice in
        # the live autograd computation nodes for the current chunk (start_pos to end_pos).
        if torch.is_grad_enabled():
            k_full = k_full.detach().clone()
            k_full[:, :, start_pos:end_pos] = _k_to_store

            v_full = v_full.detach().clone()
            v_full[:, :, start_pos:end_pos] = _v_to_store

        # Expand KV heads to match Q heads (GQA)
        k_full = k_full.repeat_interleave(self.num_queries_per_kv, dim=1)
        v_full = v_full.repeat_interleave(self.num_queries_per_kv, dim=1)


        # Rectangular causal mask (T,S)
        attn_mask = None
        if mask:
            attn_mask = self._get_or_create_kv_mask(T, end_pos, start_pos, device=x.device)

        # SDPA
        context = _run_sdpa(
            q,
            k_full,
            v_full,
            attn_mask = attn_mask,
            dropout_p = self.dropout_p
        )

        # Merge heads
        out = context.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(out)