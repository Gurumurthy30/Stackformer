"""Google Gemma decoder-only causal language model implementations.

Rebuilds Google Gemma-1 (2B and 7B) architectures on top of StackFormer block builders.

Paper reference:
    Gemma: Open Models Based on Gemini Research and Technology
    https://arxiv.org/abs/2403.08295
"""

from __future__ import annotations

import torch
import torch.nn as nn

from stackformer.generate import text_generate
from stackformer.modules.layer import BlockConfig, TransformerEncoder


class Gemma1_2B(nn.Module):
    """Gemma 1 2B style decoder-only causal language model.

    Simple explanation:
        A GPT-like text generation model that predicts the next token from left
        to right. Uses Multi-Query Attention (MQA) with Rotary Position Embeddings (RoPE)
        to reduce the KV cache footprint during autoregressive inference.

    Architecture details (current implementation):
        - Task: causal language modeling
        - Attention: Multi-Query Attention (MQA) with RoPE
        - Masking: causal mask
        - Positional encoding: RoPE (rotary positional embeddings)
        - Feed-forward: GeGLU FFN
        - Normalization: Pre-Norm RMSNorm in every block
        - Head: Linear vocabulary head

    Historical context:
        - Gemma is Google's open weights model family based on Gemini technology (2024).
        - The 2B variant employs MQA (one shared KV head) to optimize memory bandwidth
          and allow large-batch inference on edge devices.

    Paper reference:
        - Gemma paper: https://arxiv.org/abs/2403.08295

    Example:
        >>> import torch
        >>> from stackformer.models import Gemma_1_2B
        >>> model = Gemma_1_2B(vocab_size=32000, num_layers=4, embed_dim=512, num_heads=8, seq_len=128)
        >>> input_ids = torch.randint(0, 32000, (2, 64))
        >>> logits = model(input_ids)
        >>> logits.shape
        torch.Size([2, 64, 32000])

    Args:
        vocab_size (int): Size of token vocabulary.
        num_layers (int): Number of transformer decoder layers.
        embed_dim (int): Hidden embedding dimension.
        num_heads (int): Number of query attention heads.
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
            attention="mqa_rope",
            ffn="geglu",
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
        x = self.embedding(x) * (self.embedding.embedding_dim**0.5)  # (B, T, C) scale for stability
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


class Gemma1_7B(nn.Module):
    """Gemma 1 7B style decoder-only causal language model.

    Simple explanation:
        Larger Gemma model with full Multi-Head Attention (MHA) and Rotary Position
        Embeddings (RoPE). Every head maintains independent Key and Value projections,
        providing high model capacity for complex tasks.

    Architecture details (current implementation):
        - Task: causal language modeling
        - Attention: Multi-Head Attention (MHA) with RoPE
        - Masking: causal mask
        - Positional encoding: RoPE (rotary positional embeddings)
        - Feed-forward: GeGLU FFN
        - Normalization: Pre-Norm RMSNorm in every block
        - Head: Linear vocabulary head

    Historical context:
        - 7B parameter variant of Google's Gemma model family (2024).
        - Employs full MHA to maximize representational capacity across all heads.

    Paper reference:
        - Gemma paper: https://arxiv.org/abs/2403.08295

    Example:
        >>> import torch
        >>> from stackformer.models import Gemma_1_7B
        >>> model = Gemma_1_7B(vocab_size=32000, num_layers=6, embed_dim=768, num_heads=12, seq_len=128)
        >>> input_ids = torch.randint(0, 32000, (1, 32))
        >>> logits = model(input_ids)
        >>> logits.shape
        torch.Size([1, 32, 32000])

    Args:
        vocab_size (int): Size of token vocabulary.
        num_layers (int): Number of transformer decoder layers.
        embed_dim (int): Hidden embedding dimension.
        num_heads (int): Number of attention heads.
        seq_len (int): Maximum sequence context length.
        dropout (float, default=0.0): Dropout probability.
        hidden_dim (int, default=0): FFN hidden dimension (0 defaults to 4 * embed_dim).
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
            ffn="geglu",
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
        x = self.embedding(x) * (self.embedding.embedding_dim**0.5)  # (B, T, C) scale for stability
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


# Backward compatibility aliases
gemma_1_2b = Gemma1_2B
gemma_1_7b = Gemma1_7B
Gemma_1_2B = Gemma1_2B
Gemma_1_7B = Gemma1_7B