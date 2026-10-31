# Example: Building a Custom Transformer

This example demonstrates how to compose a modern custom decoder-only Transformer from scratch using StackFormer's `BlockConfig` and `TransformerEncoder` components.

We will combine:
- **Grouped-Query Attention with RoPE** (`"gqa_rope"`)
- **SwiGLU Feed-Forward Networks** (`"swiglu"`)
- **Root Mean Square Normalization** (`"rmsnorm"`)
- **Pre-Norm Architecture** (`pre_norm=True`)
- **Causal Attention Masking** (`mask_type=["causal"]`)

---

## Complete Runnable Script

```python
import torch
import torch.nn as nn
from stackformer.modules.layer import BlockConfig, TransformerEncoder

class CustomModernTransformer(nn.Module):
    """Custom decoder-only Transformer with GQA, RoPE, SwiGLU, and RMSNorm."""

    def __init__(
        self,
        vocab_size: int = 32000,
        num_layers: int = 8,
        embed_dim: int = 512,
        num_heads: int = 8,
        num_kv_heads: int = 2,
        seq_len: int = 1024,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.seq_len = seq_len

        # 1. Declaratively configure layer specifications
        cfg = BlockConfig(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            attention="gqa_rope",
            ffn="swiglu",
            norm="rmsnorm",
            pre_norm=True,
            dropout=dropout,
            mask_type=["causal"],
            qkv_bias=False,
        )

        # 2. Token Embedding
        self.tok_embeddings = nn.Embedding(vocab_size, embed_dim)

        # 3. Stacked Transformer Backbone
        # Note: RoPE is handled dynamically inside attention, so pos_embedding is None
        self.backbone = TransformerEncoder(
            cfg,
            num_layers=num_layers,
            pos_embedding=None,
            max_seq_len=seq_len,
        )

        # 4. Final Normalization and Language Modeling Head
        self.norm = nn.RMSNorm(embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)

        # Tie weights between token embedding and LM head
        self.lm_head.weight = self.tok_embeddings.weight

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # input_ids: (Batch, Seq_Len)
        x = self.tok_embeddings(input_ids)      # (B, T, C)
        x = self.backbone(x, mask=True)         # (B, T, C) with causal mask
        x = self.norm(x)                        # (B, T, C)
        logits = self.lm_head(x)                # (B, T, V)
        return logits


if __name__ == "__main__":
    # Test model initialization
    model = CustomModernTransformer(
        vocab_size=10000,
        num_layers=4,
        embed_dim=256,
        num_heads=8,
        num_kv_heads=2,
        seq_len=256,
    )

    batch_size, seq_length = 2, 64
    dummy_input = torch.randint(0, 10000, (batch_size, seq_length))

    logits = model(dummy_input)
    print("Logits shape:", logits.shape)
    assert logits.shape == (batch_size, seq_length, 10000)

    # Calculate parameter count
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total Model Parameters: {total_params / 1e6:.2f}M")
```

---

## Explanation

1. **`BlockConfig` Specification**: Passing `attention="gqa_rope"` provisions 8 query heads and 2 key/value heads with rotary embeddings calculated on the fly.
2. **`pos_embedding=None`**: When using RoPE, positional information is rotated directly into queries and keys at each layer. Additive positional embeddings on the input tokens are omitted.
3. **Weight Tying**: Tying `self.lm_head.weight` to `self.tok_embeddings.weight` reduces overall memory consumption and accelerates convergence.

