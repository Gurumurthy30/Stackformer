"""RoBERTa encoder-only language model implementation.

Paper reference:
    RoBERTa: A Robustly Optimized BERT Pretraining Approach
    https://arxiv.org/abs/1907.11692
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from stackformer.models.base import BaseModel, resolve_model_config
from stackformer.modules.layer import BlockConfig, TransformerEncoder
from stackformer.modules.position_embedding import AbsolutePositionEmbedding


class RoBERTa(BaseModel):
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
        vocab_size (int | Any, default=50265): Size of vocabulary or ModelConfig instance.
        num_layers (int, default=12): Number of encoder layers.
        embed_dim (int, default=768): Hidden dimension size.
        num_heads (int, default=12): Number of attention heads.
        seq_len (int, default=512): Maximum sequence length.
        dropout (float, default=0.1): Dropout probability.
        hidden_dim (int, default=0): FFN hidden dimension (0 defaults to 4 * embed_dim).
        padding_idx (int, default=1): Padding token ID for offset position calculation.
        qkv_bias (bool, default=True): Enable bias in linear projections.
        eps (float, default=1e-5): LayerNorm epsilon.
        device (torch.device | str, default="cpu"): Target compute device.
        dtype (torch.dtype, default=torch.float32): Tensor data type.
        config (Any | None, default=None): Optional configuration object.
    """

    def __init__(
        self,
        vocab_size: int | Any = 50265,
        num_layers: int = 12,
        embed_dim: int = 768,
        num_heads: int = 12,
        seq_len: int = 512,
        dropout: float = 0.1,
        hidden_dim: int = 0,
        padding_idx: int = 1,
        qkv_bias: bool = True,
        eps: float = 1e-5,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
        config: Any = None,
        **kwargs: Any,
    ) -> None:
        cfg, resolved = resolve_model_config(
            first_arg=vocab_size,
            explicit_config=config,
            num_layers=num_layers,
            embed_dim=embed_dim,
            num_heads=num_heads,
            seq_len=seq_len,
            dropout=dropout,
            hidden_dim=hidden_dim,
            padding_idx=padding_idx,
            qkv_bias=qkv_bias,
            eps=eps,
            device=device,
            dtype=dtype,
            **kwargs,
        )
        resolved_config = cfg or config
        super().__init__(config=resolved_config)

        if isinstance(vocab_size, int):
            resolved["vocab_size"] = vocab_size
        vocab_size = resolved.get("vocab_size", 50265)
        num_layers = resolved.get("num_layers", num_layers)
        embed_dim = resolved.get("embed_dim", embed_dim)
        num_heads = resolved.get("num_heads", num_heads)
        seq_len = resolved.get("seq_len", seq_len)
        dropout = resolved.get("dropout", dropout)
        hidden_dim = resolved.get("hidden_dim", hidden_dim)
        padding_idx = resolved.get("padding_idx", padding_idx)
        qkv_bias = resolved.get("qkv_bias", qkv_bias)
        eps = resolved.get("eps", eps)
        device = resolved.get("device", device)
        dtype = resolved.get("dtype", dtype)

        self.seq_len = seq_len
        self.padding_idx = padding_idx

        cfg_block = BlockConfig(
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

        self.backbone = TransformerEncoder(cfg_block, num_layers=num_layers, pos_embedding=None)

        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False, device=device, dtype=dtype)
        self.lm_head.weight = self.token_embedding.weight

        self.create_weights()

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            if not module.weight.is_meta:
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None and not module.bias.is_meta:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            if not module.weight.is_meta:
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.padding_idx is not None and not module.weight.is_meta:
                with torch.no_grad():
                    module.weight[module.padding_idx].fill_(0)
        elif isinstance(module, nn.LayerNorm):
            if module.weight is not None and not module.weight.is_meta:
                nn.init.ones_(module.weight)
            if module.bias is not None and not module.bias.is_meta:
                nn.init.zeros_(module.bias)

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

    def get_sharding_blocks(self) -> list[nn.Module]:
        """Return repeating encoder layer blocks for sharding and checkpointing."""
        return list(self.backbone.layers)

    def get_tied_params(self) -> list[tuple[str, str]] | None:
        """Return parameter name pairs that share underlying weights."""
        return [("token_embedding.weight", "lm_head.weight")]
