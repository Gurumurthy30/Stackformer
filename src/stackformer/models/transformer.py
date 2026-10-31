"""Original Encoder-Decoder Transformer model implementation.

Rebuilds the canonical sequence-to-sequence Transformer architecture on top of StackFormer
block components (BlockConfig, TransformerEncoder, TransformerDecoder).

Paper reference:
    Attention Is All You Need (Vaswani et al., 2017)
    https://arxiv.org/abs/1706.03762
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn

from stackformer.modules.layer import BlockConfig, TransformerDecoder, TransformerEncoder


class Transformer(nn.Module):
    """Vaswani et al. style encoder-decoder Transformer.

    Simple explanation:
        This model has two parts: an encoder that reads the source sequence
        in full (bidirectional) and a decoder that generates a target sequence
        token-by-token, attending to both its own past outputs (causal
        self-attention) and the full encoder output (cross-attention).

    Architecture details (current implementation):
        - Task: sequence-to-sequence modeling (e.g. machine translation)
        - Encoder attention: Multi-Head self-attention, no causal mask
        - Decoder attention: Causal MHA self-attention + cross-attention into encoder memory
        - Positional encoding: Sinusoidal positional embeddings (fixed)
        - Feed-forward: ReLU MLP
        - Normalization: Post-LayerNorm (pre_norm=False)
        - Head: Linear projection layer to target vocabulary

    Historical context:
        - Introduced in 2017 ("Attention Is All You Need", Vaswani et al.).
        - Foundation paper that introduced self-attention as a replacement for recurrence.

    Paper reference:
        - Transformer paper: https://arxiv.org/abs/1706.03762

    Example:
        >>> import torch
        >>> from stackformer.models import Transformer
        >>> model = Transformer(
        ...     vocab_size=32000,
        ...     embed_dim=512,
        ...     num_heads=8,
        ...     dropout=0.1,
        ...     hidden_dim=2048,
        ...     encoder_layers=6,
        ...     decoder_layers=6,
        ...     seq_len=128,
        ... )
        >>> src = torch.randint(0, 32000, (2, 40))
        >>> tgt = torch.randint(0, 32000, (2, 30))
        >>> logits = model(src, tgt)
        >>> logits.shape
        torch.Size([2, 30, 32000])

    Args:
        vocab_size (int): Token vocabulary size.
        embed_dim (int): Model embedding dimension.
        num_heads (int): Number of attention heads.
        dropout (float, default=0.1): Dropout probability.
        hidden_dim (int, default=0): Inner hidden FFN dimension (0 defaults to 4 * embed_dim).
        encoder_layers (int, default=6): Number of encoder layers.
        decoder_layers (int, default=6): Number of decoder layers.
        seq_len (int, default=512): Maximum context sequence length.
        eps (float, default=1e-5): LayerNorm epsilon.
        device (torch.device | str, default="cpu"): Compute device.
        dtype (torch.dtype, default=torch.float32): Model data type.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.1,
        hidden_dim: int = 0,
        encoder_layers: int = 6,
        decoder_layers: int = 6,
        seq_len: int = 512,
        eps: float = 1e-5,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.embed_scale = math.sqrt(embed_dim)

        cfg = BlockConfig(
            embed_dim=embed_dim,
            num_heads=num_heads,
            hidden_dim=hidden_dim,
            attention="mha",
            ffn="relu",
            norm="layernorm",
            pre_norm=False,
            eps=eps,
            dropout=dropout,
            device=device,
            dtype=dtype,
        )

        self.token_emb = nn.Embedding(vocab_size, embed_dim, device=device, dtype=dtype)

        self.encoder_stack = TransformerEncoder(
            cfg,
            num_layers=encoder_layers,
            pos_embedding="sinusoidal",
            max_seq_len=seq_len,
        )

        self.decoder_stack = TransformerDecoder(
            cfg,
            num_layers=decoder_layers,
            pos_embedding="sinusoidal",
            max_seq_len=seq_len,
        )

        self.out_proj = nn.Linear(embed_dim, vocab_size, device=device, dtype=dtype)

    def encode(self, src: torch.Tensor) -> torch.Tensor:
        """Encode source sequence into hidden representations tensor.

        Args:
            src (torch.Tensor): Source token IDs tensor of shape ``(B, S)``.

        Returns:
            torch.Tensor: Encoder memory tensor of shape ``(B, S, C)``.
        """
        x = self.token_emb(src) * self.embed_scale  # (B, S, C)
        return self.encoder_stack(x, mask=False)  # (B, S, C)

    def decode(self, tgt: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        """Decode target sequence given encoder memory representations.

        Args:
            tgt (torch.Tensor): Target token IDs tensor of shape ``(B, T)``.
            memory (torch.Tensor): Encoder memory tensor of shape ``(B, S, C)``.

        Returns:
            torch.Tensor: Decoder output representations of shape ``(B, T, C)``.
        """
        x = self.token_emb(tgt) * self.embed_scale  # (B, T, C)
        return self.decoder_stack(x, memory)  # (B, T, C)

    def forward(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Forward pass for sequence-to-sequence Transformer.

        Args:
            source (torch.Tensor): Source sequence token IDs tensor of shape ``(B, S)``.
            target (torch.Tensor): Target sequence token IDs tensor of shape ``(B, T)``.

        Returns:
            torch.Tensor: Target vocabulary logits tensor of shape ``(B, T, V)``.
        """
        memory = self.encode(source)  # (B, S, C)
        out = self.decode(target, memory)  # (B, T, C)
        return self.out_proj(out)  # (B, T, V)