"""Meta LLaMA decoder-only causal language model implementations.

Provides LLaMA 1 (MHA + RoPE + SwiGLU) and LLaMA 2 (GQA + stateful KV cache + RoPE + SwiGLU)
architectures built on StackFormer block components.

Paper references:
    LLaMA 1: LLaMA: Open and Efficient Foundation Language Models
    https://arxiv.org/abs/2302.13971

    LLaMA 2: LLaMA 2: Open Foundation and Fine-Tuned Chat Models
    https://arxiv.org/abs/2307.09288
"""

from __future__ import annotations

import torch
import torch.nn as nn

from stackformer.generate import text_generate
from stackformer.modules.Attention import kv_cache_group_query
from stackformer.modules.layer import BlockConfig, TransformerEncoder, _build_ffn, _build_norm


class Llama1(nn.Module):
    """LLaMA 1 style decoder-only causal language model.

    Simple explanation:
        Reads token sequences and predicts the next token using only left
        context (causal decoding). Designed for autoregressive text generation
        with RoPE position encoding and SwiGLU feed-forward blocks.

    Architecture details (current implementation):
        - Task: causal language modeling
        - Attention: Multi-Head Attention (MHA) with RoPE
        - Masking: causal mask
        - Positional encoding: RoPE (rotary positional embeddings)
        - Feed-forward: SwiGLU FFN
        - Normalization: Pre-Norm RMSNorm in every block
        - Head: Linear vocabulary head

    Historical context:
        - Introduced by Meta AI in 2023 (Touvron et al.).
        - Proved that training smaller models on significantly more tokens yields superior
          inference efficiency without sacrificing benchmark performance.

    Paper reference:
        - LLaMA 1 paper: https://arxiv.org/abs/2302.13971

    Example:
        >>> import torch
        >>> from stackformer.models import Llama_1
        >>> model = Llama_1(vocab_size=32000, num_layers=4, embed_dim=512, num_heads=8, seq_len=128)
        >>> input_ids = torch.randint(0, 32000, (2, 64))
        >>> logits = model(input_ids)
        >>> logits.shape
        torch.Size([2, 64, 32000])

    Args:
        vocab_size (int): Size of token vocabulary.
        num_layers (int): Number of transformer decoder layers.
        embed_dim (int): Hidden embedding dimension.
        num_heads (int): Number of attention heads.
        seq_len (int): Maximum sequence context length.
        dropout (float, default=0.0): Dropout probability.
        hidden_dim (int, default=0): Inner hidden FFN dimension (0 defaults to 4 * embed_dim).
        eps (float, default=1e-5): RMSNorm epsilon.
        device (torch.device | str, default="cpu"): Target compute device.
        dtype (torch.dtype, default=torch.float32): Tensor data type.
    """

    def __init__(
        self,
        vocab_size: int,
        num_layers: int,
        embed_dim: int,
        num_heads: int,
        seq_len: int,
        dropout: float = 0.0,
        hidden_dim: int = 0,
        eps: float = 1e-5,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.seq_len = seq_len

        cfg = BlockConfig(
            embed_dim=embed_dim,
            num_heads=num_heads,
            hidden_dim=hidden_dim,
            attention="mha_rope",
            ffn="swiglu",
            norm="rmsnorm",
            pre_norm=True,
            dropout=dropout,
            device=device,
            dtype=dtype,
        )

        self.embedding = nn.Embedding(vocab_size, embed_dim, dtype=dtype, device=device)
        self.backbone = TransformerEncoder(cfg, num_layers=num_layers, pos_embedding=None)
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False, dtype=dtype, device=device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T)
        x = self.embedding(x)  # (B, T, C)
        x = self.backbone(x, mask=True)  # (B, T, C) causal mask
        return self.lm_head(x)  # (B, T, V)

    @torch.no_grad()
    def generate(
        self,
        prompt_ids: torch.Tensor,
        max_context_len: int = 128,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float = 1.0,
        eos_token_id: int | None = None,
    ) -> torch.Tensor:
        """Autoregressively generate tokens from prompt token IDs.

        Args:
            prompt_ids (torch.Tensor): Prompt token IDs tensor of shape ``(B, T)`` or ``(T,)``.
            max_context_len (int, default=128): Maximum context length window.
            max_new_tokens (int, default=50): Number of tokens to generate.
            temperature (float, default=1.0): Temperature sampling parameter.
            top_k (int | None, default=None): Top-k filtering threshold.
            top_p (float, default=1.0): Nucleus top-p filtering threshold.
            eos_token_id (int | None, default=None): Optional EOS token ID.

        Returns:
            torch.Tensor: Generated token IDs of shape ``(B, T + num_generated)``.
        """
        return text_generate(
            self,
            prompt_ids,
            max_context_len=max_context_len,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            eos_token_id=eos_token_id,
        )


class _Llama2Block(nn.Module):
    """Single LLaMA 2 decoder block with GQA and stateful KV cache.

    Constructor args:
        cfg (BlockConfig): Configuration instance.
        batch_size (int): Pre-allocated cache batch size.
        kv_seq_len (int): Pre-allocated cache sequence length.

    Forward args:
        x (torch.Tensor): Input token features of shape ``(B, T, C)``.
        start_pos (int): Starting position offset in KV cache.

    Returns:
        torch.Tensor: Output token features of shape ``(B, T, C)``.
    """

    def __init__(self, cfg: BlockConfig, batch_size: int, kv_seq_len: int) -> None:
        super().__init__()
        self.norm1 = _build_norm(cfg)
        self.attn = kv_cache_group_query(
            embed_dim=cfg.embed_dim,
            num_query_heads=cfg.num_heads,
            num_kv_heads=cfg.num_kv_heads,
            batch_size=batch_size,
            kv_seq_len=kv_seq_len,
            dropout=cfg.dropout,
            device=cfg.device,
            dtype=cfg.dtype,
        )
        self.norm2 = _build_norm(cfg)
        self.ffn = _build_ffn(cfg)

    def forward(self, x: torch.Tensor, start_pos: int) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), start_pos, rope=True)
        x = x + self.ffn(self.norm2(x))
        return x


class Llama2(nn.Module):
    """LLaMA 2 style decoder-only causal language model with GQA and persistent KV cache.

    Simple explanation:
        Autoregressively generates text using Grouped-Query Attention (GQA) and persistent
        KV caching. The stateful cache allows fast generation by reusing previous Key and Value
        tensors across decoding steps.

    Architecture details (current implementation):
        - Task: causal language modeling
        - Attention: Grouped-Query Attention (GQA) with stateful KV cache
        - Masking: causal mask
        - Positional encoding: RoPE (rotary positional embeddings)
        - Feed-forward: SwiGLU FFN
        - Normalization: Pre-Norm RMSNorm in every block
        - Head: Linear vocabulary head

    Historical context:
        - Introduced by Meta AI in 2023 (Touvron et al.).
        - Introduced Grouped-Query Attention to scale sequence length efficiently.

    Paper reference:
        - LLaMA 2 paper: https://arxiv.org/abs/2307.09288

    Example:
        >>> import torch
        >>> from stackformer.models import Llama_2
        >>> model = Llama_2(
        ...     vocab_size=32000,
        ...     num_layers=4,
        ...     embed_dim=512,
        ...     num_query_heads=8,
        ...     num_kv_heads=2,
        ...     batch_size=1,
        ...     kv_seq_len=128,
        ... )
        >>> input_ids = torch.randint(0, 32000, (1, 16))
        >>> logits = model(input_ids, start_pos=0)
        >>> logits.shape
        torch.Size([1, 16, 32000])

    Args:
        vocab_size (int): Size of token vocabulary.
        num_layers (int): Number of transformer layers.
        embed_dim (int): Hidden state embedding dimension.
        num_query_heads (int): Number of Query attention heads.
        num_kv_heads (int): Number of Key/Value attention heads.
        batch_size (int): Pre-allocated batch size for KV cache tensor.
        kv_seq_len (int): Pre-allocated sequence length limit for KV cache tensor.
        hidden_dim (int, default=0): Inner hidden FFN dimension (0 defaults to 8/3 expansion).
        dropout (float, default=0.0): Dropout probability.
        eps (float, default=1e-5): RMSNorm epsilon.
        dtype (torch.dtype, default=torch.float32): Tensor data type.
        device (torch.device | str, default="cpu"): Target compute device.
    """

    def __init__(
        self,
        vocab_size: int,
        num_layers: int,
        embed_dim: int,
        num_query_heads: int,
        num_kv_heads: int,
        batch_size: int,
        kv_seq_len: int,
        hidden_dim: int = 0,
        dropout: float = 0.0,
        eps: float = 1e-5,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str = "cpu",
    ) -> None:
        super().__init__()
        assert embed_dim % num_query_heads == 0, "embed_dim must be divisible by num_query_heads"
        assert num_query_heads % num_kv_heads == 0, "num_query_heads must be divisible by num_kv_heads"

        self.seq_len = kv_seq_len

        if hidden_dim == 0:
            hidden_dim = int(int(8 / 3 * embed_dim) / 256) * 256

        cfg = BlockConfig(
            embed_dim=embed_dim,
            num_heads=num_query_heads,
            num_kv_heads=num_kv_heads,
            hidden_dim=hidden_dim,
            attention="gqa_rope",
            ffn="swiglu",
            norm="rmsnorm",
            pre_norm=True,
            dropout=dropout,
            device=device,
            dtype=dtype,
        )

        self.embedding = nn.Embedding(vocab_size, embed_dim, dtype=dtype, device=device)

        self.layers = nn.ModuleList([
            _Llama2Block(cfg, batch_size=batch_size, kv_seq_len=kv_seq_len)
            for _ in range(num_layers)
        ])

        self.final_norm = _build_norm(cfg)
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False, dtype=dtype, device=device)

    def forward(self, input_ids: torch.Tensor, start_pos: int = 0) -> torch.Tensor:
        # input_ids: (B, T)
        x = self.embedding(input_ids)  # (B, T, C)
        for layer in self.layers:
            x = layer(x, start_pos)
        x = self.final_norm(x)  # (B, T, C)
        return self.lm_head(x)  # (B, T, V)

    @torch.no_grad()
    def generate(
        self,
        prompt_ids: torch.Tensor,
        max_context_len: int = 128,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float = 1.0,
        eos_token_id: int | None = None,
    ) -> torch.Tensor:
        """Autoregressively generate tokens from prompt token IDs.

        Args:
            prompt_ids (torch.Tensor): Prompt token IDs tensor of shape ``(B, T)`` or ``(T,)``.
            max_context_len (int, default=128): Maximum context length window.
            max_new_tokens (int, default=50): Number of tokens to generate.
            temperature (float, default=1.0): Temperature sampling parameter.
            top_k (int | None, default=None): Top-k filtering threshold.
            top_p (float, default=1.0): Nucleus top-p filtering threshold.
            eos_token_id (int | None, default=None): Optional EOS token ID.

        Returns:
            torch.Tensor: Generated token IDs of shape ``(B, T + num_generated)``.
        """
        return text_generate(
            self,
            prompt_ids,
            max_context_len=max_context_len,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            eos_token_id=eos_token_id,
        )
        
    def reset_cache(self):
        for layer in self.layers:
            layer.attn.reset_cache()
        
    def prefill(self, prompt_ids: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Prefill the KV-cache for models that support it.

        Args:
            prompt_ids (torch.Tensor): Prompt token IDs of shape ``(B, T)``.

        Returns:
            tuple[torch.Tensor, dict[str, torch.Tensor]]: Logits tensor of shape ``(B, T, V)`` and KV-cache dictionary.
        """
        B, T = prompt_ids.shape
        if T > self.seq_len:
            raise ValueError(
                f"Prompt length {T} exceeds KV cache capacity {self.seq_len}. "
                f"Truncate the prompt or rebuild the model with a larger kv_seq_len."
            )
        self.reset_cache()
        logits = self.forward(prompt_ids, start_pos=0)
        cache = {"start_pos": T}
        return logits, cache
    
    def decode(self, next_token: torch.Tensor, cache: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Decode the next token using the KV-cache for models that support it.

        Args:
            next_token (torch.Tensor): Next token IDs of shape ``(B, 1)``.
            cache (dict[str, torch.Tensor]): KV-cache dictionary from prefill.

        Returns:
            tuple[torch.Tensor, dict[str, torch.Tensor]]: Logits tensor of shape ``(B, 1, V)`` and updated KV-cache dictionary.
        """
        start_pos = cache["start_pos"]
        end_pos = start_pos + next_token.shape[1]
        if end_pos > self.seq_len:
            raise ValueError(
                f"KV cache capacity exceeded: start_pos={start_pos} + new_tokens={next_token.shape[1]} "
                f"= {end_pos} > cache length {self.seq_len}. Generation must stop or the model must be "
                f"rebuilt with a larger kv_seq_len."
            )
        logits = self.forward(next_token, start_pos=start_pos)
        cache["start_pos"] = end_pos
        return logits, cache


# Backward compatibility aliases
llama_1 = Llama1
llama_2 = Llama2
Llama_1 = Llama1
Llama_2 = Llama2
_llama_2_Block = _Llama2Block

