"""Encoder-decoder sequence-to-sequence model architecture specifications.

Defines base classes for sequence-to-sequence models (e.g. T5, BART) combining bidirectional
encoders with causal cross-attention decoders.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class EncoderDecoder(nn.Module):
    """Sequence-to-sequence Encoder-Decoder architecture container (e.g. T5 / BART).

    Simple explanation:
        An encoder-decoder model processes an input sequence using a bidirectional
        encoder and generates an output sequence token-by-token using a decoder with
        cross-attention into the encoder representations.

    Architecture details (current implementation):
        - Task: sequence-to-sequence (translation, summarization)
        - Attention: bidirectional self-attention in encoder, causal self-attention + cross-attention in decoder
        - Masking: bidirectional (encoder), causal (decoder)
        - Positional encoding: relative or absolute positional embeddings
        - Feed-forward: GELU / ReLU / SwiGLU FFN
        - Normalization: LayerNorm / RMSNorm
        - Head: Linear vocabulary output projection

    Paper reference:
        - T5 Paper: https://arxiv.org/abs/1910.10683

    Example:
        >>> model = EncoderDecoder(vocab_size=1000, embed_dim=128)
        >>> src = torch.randint(0, 1000, (2, 16))
        >>> tgt = torch.randint(0, 1000, (2, 8))
        >>> out = model(src, tgt)
        >>> out.shape
        torch.Size([2, 8, 1000])

    Args:
        vocab_size (int, default=32100): Size of token vocabulary.
        embed_dim (int, default=512): Dimensionality of embeddings and hidden states.
        num_encoder_layers (int, default=6): Number of encoder layers.
        num_decoder_layers (int, default=6): Number of decoder layers.
    """

    def __init__(
        self,
        vocab_size: int = 32100,
        embed_dim: int = 512,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.shared_embed = nn.Embedding(vocab_size, embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)

    def forward(self, encoder_input_ids: torch.Tensor, decoder_input_ids: torch.Tensor) -> torch.Tensor:
        # encoder_input_ids: (B, S), decoder_input_ids: (B, T)
        dec_x = self.shared_embed(decoder_input_ids)  # (B, T, C)
        # Placeholder forward pass returns linear logits over decoder tokens
        return self.lm_head(dec_x)  # (B, T, V)