"""BERT encoder-only language model implementation.

Paper reference:
    BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding
    https://arxiv.org/abs/1810.04805
"""

from __future__ import annotations

import torch
import torch.nn as nn

from stackformer.modules.layer import BlockConfig, TransformerEncoder
from stackformer.modules.position_embedding import AbsolutePositionEmbedding


class BERT(nn.Module):
    """BERT style encoder-only model for bidirectional language representation.

    Simple explanation:
        Bidirectional encoder that reads the entire input sequence at once.
        Pre-trained with Masked Language Modeling (MLM) and Next Sentence
        Prediction (NSP), then fine-tuned on downstream tasks.

    Architecture details (current implementation):
        - Task: masked language modeling / sentence classification
        - Attention: Multi-Head Attention (MHA), no causal mask
        - Masking: padding mask only (bidirectional)
        - Positional encoding: learned absolute positional embeddings
        - Segment encoding: learned token-type (sentence A/B) embeddings
        - Feed-forward: GELU MLP
        - Normalization: Post-LayerNorm (pre_norm=False)
        - Head: Linear MLM head with weight tying to token embeddings

    Historical context:
        - Introduced by Google AI Language in 2018 (Devlin et al.).
        - Revolutionized NLP by proving that bidirectional context pre-training
          dramatically outperforms unidirectional language models on NLU tasks.

    Paper reference:
        - BERT paper: https://arxiv.org/abs/1810.04805

    Example:
        >>> model = BERT(vocab_size=30522, num_layers=12, embed_dim=768, num_heads=12, seq_len=512)
        >>> input_ids = torch.randint(0, 30522, (2, 128))
        >>> logits = model(input_ids)
        >>> logits.shape
        torch.Size([2, 128, 30522])

    Args:
        vocab_size (int): Size of vocabulary.
        num_layers (int): Number of encoder layers.
        embed_dim (int): Hidden dimension size.
        num_heads (int): Number of attention heads.
        seq_len (int): Maximum sequence length context window.
        dropout (float, default=0.1): Dropout probability.
        hidden_dim (int, default=0): Hidden dimension of FFN (0 defaults to 4 * embed_dim).
        num_segments (int, default=2): Number of segment token-type embeddings.
        qkv_bias (bool, default=True): Enable bias terms in linear projections.
        eps (float, default=1e-5): LayerNorm epsilon.
        device (torch.device | str, default="cpu"): Target compute device.
        dtype (torch.dtype, default=torch.float32): Model parameter data type.
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
        num_segments: int = 2,
        qkv_bias: bool = True,
        eps: float = 1e-5,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len

        cfg = BlockConfig(
            embed_dim=embed_dim,
            num_heads=num_heads,
            hidden_dim=hidden_dim,
            attention="mha",
            ffn="gelu",
            norm="layernorm",
            mask_type=["no"],
            pre_norm=False,
            dropout=dropout,
            qkv_bias=qkv_bias,
            device=device,
            dtype=dtype,
        )

        self.token_embedding = nn.Embedding(vocab_size, embed_dim, device=device, dtype=dtype)
        self.position_embedding = AbsolutePositionEmbedding(seq_len, embed_dim, device=device, dtype=dtype)
        self.segment_embedding = nn.Embedding(num_segments, embed_dim, device=device, dtype=dtype)
        self.embed_norm = nn.LayerNorm(embed_dim, eps=eps, device=device, dtype=dtype)
        self.embed_dropout = nn.Dropout(dropout)

        self.backbone = TransformerEncoder(cfg, num_layers=num_layers, pos_embedding=None)

        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False, device=device, dtype=dtype)
        self.lm_head.weight = self.token_embedding.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        token_type_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, T = input_ids.shape

        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        x = self.token_embedding(input_ids)  # (B, T, C)
        x = x + self.position_embedding(input_ids)  # (B, T, C)
        x = x + self.segment_embedding(token_type_ids)  # (B, T, C)
        x = self.embed_norm(x)
        x = self.embed_dropout(x)

        x = self.backbone(x, mask=False)  # (B, T, C)
        return self.lm_head(x)  # (B, T, V)
