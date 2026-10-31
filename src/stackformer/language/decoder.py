"""Autoregressive decoder-only language model specifications and base components.

Defines base and placeholder classes for decoder-only architectures (e.g. GPT, Mistral, LLaMA).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class Decoder(nn.Module):
    """Base abstract decoder module for autoregressive language modeling.

    Constructor args:
        vocab_size (int, default=32000): Vocabulary size.
        embed_dim (int, default=4096): Hidden state dimension.
        num_layers (int, default=32): Number of decoder transformer layers.

    Forward args:
        input_ids (torch.Tensor): Token ID tensor of shape ``(B, T)``.

    Returns:
        torch.Tensor: Logits tensor of shape ``(B, T, V)``.

    Example:
        >>> decoder = Decoder(vocab_size=1000, embed_dim=128, num_layers=2)
        >>> x = torch.randint(0, 1000, (2, 10))
        >>> out = decoder(x)
        >>> out.shape
        torch.Size([2, 10, 1000])
    """

    def __init__(self, vocab_size: int = 32000, embed_dim: int = 4096, num_layers: int = 32) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # input_ids: (B, T)
        x = self.token_embedding(input_ids)  # (B, T, C)
        return self.head(x)  # (B, T, V)


class GPTDecoder(Decoder):
    """Decoder-only model architecture specification for GPT-style models.

    Args:
        vocab_size (int, default=50257): Token vocabulary size.
        embed_dim (int, default=768): Model embedding dimension.
        num_layers (int, default=12): Number of decoder block layers.
    """

    pass