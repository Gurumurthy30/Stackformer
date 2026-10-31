"""RoBERTa encoder-only language model implementation.

Paper reference:
    RoBERTa: A Robustly Optimized BERT Pretraining Approach
    https://arxiv.org/abs/1907.11692
"""

from __future__ import annotations

import torch
import torch.nn as nn

from stackformer.modules.layer import BlockConfig, TransformerEncoder
from stackformer.modules.position_embedding import AbsolutePositionEmbedding


class RoBERTa(nn.Module):
    """RoBERTa style encoder-only model.

    Simple explanation:
        A robustly optimized BERT architecture. Removes Next Sentence Prediction (NSP),
        uses dynamic masking, trains longer with larger batch sizes, and eliminates segment embeddings.

    Architecture details (current implementation):
        - Task: masked language modeling
        - Attention: Multi-Head Attention (MHA), no causal mask
        - Masking: padding mask only (bidirectional)
        - Positional encoding: learned absolute position embeddings offset by padding_idx
        - Segment encoding: none (removed NSP task)
        - Feed-forward: GELU MLP
        - Normalization: Post-LayerNorm (pre_norm=False)
        - Head: Linear MLM head tied to token embeddings

    Historical context:
        - Introduced by Meta AI in 2019 (Liu et al.).
        - Showed that original BERT was significantly undertrained and that removing NSP
          with longer pre-training yields superior performance.

    Paper reference:
        - RoBERTa paper: https://arxiv.org/abs/1907.11692

    Example:
        >>> model = RoBERTa(vocab_size=50265, num_layers=12, embed_dim=768, num_heads=12, seq_len=512)
        >>> input_ids = torch.randint(0, 50265, (2, 128))
        >>> logits = model(input_ids)
        >>> logits.shape
        torch.Size([2, 128, 50265])

    Args:
        vocab_size (int): Size of vocabulary.
        num_layers (int): Number of encoder layers.
        embed_dim (int): Hidden dimension size.
        num_heads (int): Number of attention heads.
        seq_len (int): Maximum sequence length.
        dropout (float, default=0.1): Dropout probability.
        hidden_dim (int, default=0): FFN hidden dimension (0 defaults to 4 * embed_dim).
        padding_idx (int, default=1): Padding token ID for offset position calculation.
        qkv_bias (bool, default=True): Enable bias in linear projections.
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
        padding_idx: int = 1,
        qkv_bias: bool = True,
        eps: float = 1e-5,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.padding_idx = padding_idx

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

        self.token_embedding = nn.Embedding(
            vocab_size,
            embed_dim,
            padding_idx=padding_idx,
            device=device,
            dtype=dtype,
        )
        self.position_embedding = AbsolutePositionEmbedding(
            seq_len + padding_idx, embed_dim, device=device, dtype=dtype
        )

        self.embed_norm = nn.LayerNorm(embed_dim, eps=eps, device=device, dtype=dtype)
        self.embed_dropout = nn.Dropout(dropout)

        self.backbone = TransformerEncoder(cfg, num_layers=num_layers, pos_embedding=None)

        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False, device=device, dtype=dtype)
        self.lm_head.weight = self.token_embedding.weight

    def _create_position_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Compute Fairseq-style offset position IDs starting from padding_idx + 1."""
        mask = input_ids.ne(self.padding_idx).int()
        return (mask.cumsum(dim=-1) + self.padding_idx) * mask + self.padding_idx * (1 - mask)

    def forward(
        self,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        position_ids = self._create_position_ids(input_ids)  # (B, T)

        x = self.token_embedding(input_ids)  # (B, T, C)
        x = x + self.position_embedding(position_ids)  # (B, T, C)
        x = self.embed_norm(x)
        x = self.embed_dropout(x)

        x = self.backbone(x, mask=False)  # (B, T, C)
        return self.lm_head(x)  # (B, T, V)
