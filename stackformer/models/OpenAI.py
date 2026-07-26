"""OpenAI GPT-family decoder-only language model implementations.

Provides GPT-1 (Post-LayerNorm) and GPT-2 (Pre-LayerNorm) causal language models built on
StackFormer block components.

Paper references:
    GPT-1: Improving Language Understanding by Generative Pre-Training (2018)
    https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf

    GPT-2: Language Models are Unsupervised Multitask Learners (2019)
    https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf
"""

from __future__ import annotations

import torch
import torch.nn as nn

from stackformer.generate import text_generate
from stackformer.modules.layer import BlockConfig, TransformerEncoder
from stackformer.modules.position_embedding import AbsolutePositionEmbedding


class GPT_1(nn.Module):
    """GPT-1 style decoder-only causal language model.

    Simple explanation:
        GPT-1 is an early large-scale autoregressively trained transformer that predicts
        the next token given previous tokens. It uses Post-LayerNorm, learned
        absolute position embeddings, and GELU feed-forward blocks.

    Architecture details (current implementation):
        - Task: causal language modeling
        - Attention: Multi-Head Attention (MHA), causal mask
        - Positional encoding: learned absolute positional embeddings
        - Feed-forward: GELU FFN
        - Normalization: Post-LayerNorm (pre_norm=False)
        - Head: Linear vocabulary head

    Historical context:
        - Introduced by OpenAI in 2018 (Radford et al.).
        - Established generative pre-training followed by discriminative fine-tuning
          as a core paradigm in NLP.

    Paper reference:
        - GPT-1 paper: https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf

    Example:
        >>> import torch
        >>> from stackformer.models import GPT_1
        >>> model = GPT_1(vocab_size=50257, num_layers=4, embed_dim=512, num_heads=8, seq_len=128)
        >>> x = torch.randint(0, 50257, (2, 32))
        >>> logits = model(x)
        >>> logits.shape
        torch.Size([2, 32, 50257])

    Args:
        vocab_size (int): Size of token vocabulary.
        num_layers (int): Number of transformer decoder layers.
        embed_dim (int): Hidden embedding dimension.
        num_heads (int): Number of attention heads.
        seq_len (int): Maximum sequence context length.
        dropout (float, default=0.1): Dropout probability.
        hidden_dim (int, default=0): Inner hidden FFN dimension (0 defaults to 4 * embed_dim).
        qkv_bias (bool, default=True): Enable bias terms in linear projections.
        eps (float, default=1e-5): LayerNorm epsilon.
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
        dropout: float = 0.1,
        hidden_dim: int = 0,
        qkv_bias: bool = True,
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
            attention="mha",
            ffn="gelu",
            norm="layernorm",
            pre_norm=False,
            dropout=dropout,
            qkv_bias=qkv_bias,
            device=device,
            dtype=dtype,
        )

        self.embedding = nn.Embedding(vocab_size, embed_dim, device=device, dtype=dtype)
        self.position_embedding = AbsolutePositionEmbedding(seq_len, embed_dim, device=device, dtype=dtype)

        self.backbone = TransformerEncoder(cfg, num_layers=num_layers, pos_embedding=None)
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T)
        x = self.embedding(x) + self.position_embedding(x)  # (B, T, C)
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

class GPT_2(nn.Module):
    """GPT-2 style decoder-only causal language model.

    Simple explanation:
        GPT-2 scales the decoder-only transformer and switches to Pre-LayerNorm
        for superior gradient flow in deep stacks. Otherwise identical to GPT-1.

    Architecture details (current implementation):
        - Task: causal language modeling
        - Attention: Multi-Head Attention (MHA), causal mask
        - Positional encoding: learned absolute positional embeddings
        - Feed-forward: GELU FFN
        - Normalization: Pre-LayerNorm (pre_norm=True) + final LayerNorm
        - Head: Linear vocabulary head

    Historical context:
        - Introduced by OpenAI in 2019 (Radford et al.).
        - Demonstrated strong zero-shot and few-shot generation capabilities via language model scaling.

    Paper reference:
        - GPT-2 paper: https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf

    Example:
        >>> import torch
        >>> from stackformer.models import GPT_2
        >>> model = GPT_2(vocab_size=50257, num_layers=6, embed_dim=768, num_heads=12, seq_len=128)
        >>> x = torch.randint(0, 50257, (1, 24))
        >>> logits = model(x)
        >>> logits.shape
        torch.Size([1, 24, 50257])

    Args:
        vocab_size (int): Size of token vocabulary.
        num_layers (int): Number of transformer decoder layers.
        embed_dim (int): Hidden embedding dimension.
        num_heads (int): Number of attention heads.
        seq_len (int): Maximum sequence context length.
        dropout (float, default=0.1): Dropout probability.
        hidden_dim (int, default=0): Inner hidden FFN dimension (0 defaults to 4 * embed_dim).
        qkv_bias (bool, default=True): Enable bias terms in linear projections.
        eps (float, default=1e-5): LayerNorm epsilon.
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
        dropout: float = 0.1,
        hidden_dim: int = 0,
        qkv_bias: bool = True,
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
            attention="mha",
            ffn="gelu",
            norm="layernorm",
            pre_norm=True,
            dropout=dropout,
            qkv_bias=qkv_bias,
            device=device,
            dtype=dtype,
        )

        self.embedding = nn.Embedding(vocab_size, embed_dim, device=device, dtype=dtype)
        self.position_embedding = AbsolutePositionEmbedding(seq_len, embed_dim, device=device, dtype=dtype)

        self.backbone = TransformerEncoder(cfg, num_layers=num_layers, pos_embedding=None)
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T)
        x = self.embedding(x) + self.position_embedding(x)  # (B, T, C)
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
