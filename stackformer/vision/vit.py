"""Vision Transformer (ViT) architecture implementation for image classification.

Implements patch projection tokenization, standard transformer encoder blocks with Pre-Norm,
learnable absolute positional embeddings, and classification head.

Paper reference:
    An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale
    https://arxiv.org/abs/2010.11929
"""

from __future__ import annotations

import torch
import torch.nn as nn

from stackformer.modules.Attention import MultiHeadAttention
from stackformer.modules.Feed_forward import FeedForwardGELU


class PatchEmbedding(nn.Module):
    """Convolutional patch embedding layer for Vision Transformers.

    Constructor args:
        img_size (int, default=224): Height and width of input image.
        patch_size (int, default=16): Height and width of image patches.
        embed_dim (int, default=768): Dimension of patch projection vectors.
        in_channels (int, default=3): Number of input image channels.

    Learnable parameters:
        proj.weight: Shape ``(embed_dim, in_channels, patch_size, patch_size)``. Convolutional kernel.
        proj.bias: Shape ``(embed_dim,)``. Bias vector.

    Forward args:
        x (torch.Tensor): Input images tensor of shape ``(B, C, H, W)``.

    Returns:
        torch.Tensor: Flattened patch tokens of shape ``(B, N, C)`` where N = (H/P) * (W/P).

    Rules:
        - Image size H and W must be divisible by patch_size.

    Example:
        >>> patch_embed = PatchEmbedding(img_size=224, patch_size=16, embed_dim=768)
        >>> x = torch.randn(2, 3, 224, 224)
        >>> out = patch_embed(x)
        >>> out.shape
        torch.Size([2, 196, 768])
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        embed_dim: int = 768,
        in_channels: int = 3,
        Emb_dim: int | None = None,
    ) -> None:
        super().__init__()
        if Emb_dim is not None:
            embed_dim = Emb_dim
        assert img_size % patch_size == 0, "Image size must be divisible by patch size"
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.Emb_dim = embed_dim
        self.num_patches = (img_size // patch_size) ** 2

        self.projection = nn.Conv2d(
            in_channels, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W) -> (B, N, C)
        x = self.projection(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class Block(nn.Module):
    """Transformer block combining pre-norm multi-head self-attention and GELU FFN.

    Constructor args:
        embed_dim (int): Embedding dimension.
        num_heads (int): Number of attention heads.
        dropout (float): Dropout probability.
        hidden_dim (int): Inner hidden dimension of feed-forward layer.
        eps (float, default=1e-5): LayerNorm epsilon.
        device (torch.device | str | None, default=None): Target device.
        dtype (torch.dtype, default=torch.float32): Tensor data type.
        Emb_dim (int | None, default=None): Deprecated alias for embed_dim.

    Forward args:
        x (torch.Tensor): Input token embeddings tensor of shape ``(B, N, C)``.

    Returns:
        torch.Tensor: Transformed token embeddings tensor of shape ``(B, N, C)``.

    Example:
        >>> block = Block(embed_dim=768, num_heads=12, dropout=0.1, hidden_dim=3072)
        >>> x = torch.randn(2, 197, 768)
        >>> out = block(x)
        >>> out.shape
        torch.Size([2, 197, 768])
    """

    def __init__(
        self,
        embed_dim: int = 768,
        num_heads: int = 12,
        dropout: float = 0.1,
        hidden_dim: int = 3072,
        eps: float = 1e-5,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
        Emb_dim: int | None = None,
    ) -> None:
        super().__init__()
        if Emb_dim is not None:
            embed_dim = Emb_dim
        self.attention = MultiHeadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            qkv_bias=True,
            device=device,
            dtype=dtype,
        )
        self.norm1 = nn.LayerNorm(embed_dim, eps=eps, dtype=dtype)
        self.ff_gelu = FeedForwardGELU(embed_dim, hidden_dim, dropout, device=device, dtype=dtype)
        self.norm2 = nn.LayerNorm(embed_dim, eps=eps, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-norm attention
        residual = x
        x = self.norm1(x)
        x = self.attention(x, mask=False)
        x = x + residual

        # Pre-norm FFN
        residual = x
        x = self.norm2(x)
        x = self.ff_gelu(x)
        x = x + residual

        return x


class Encoder(nn.Module):
    """Stack of Vision Transformer encoder blocks.

    Constructor args:
        num_layers (int): Number of encoder blocks.
        embed_dim (int): Embedding dimension.
        num_heads (int): Number of attention heads.
        dropout (float): Dropout probability.
        hidden_dim (int): Feed-forward inner hidden dimension.
        eps (float, default=1e-5): LayerNorm epsilon.
        device (torch.device | str | None, default=None): Target device.
        dtype (torch.dtype, default=torch.float32): Tensor data type.
        Emb_dim (int | None, default=None): Deprecated alias for embed_dim.

    Forward args:
        x (torch.Tensor): Input token embeddings tensor of shape ``(B, N, C)``.

    Returns:
        torch.Tensor: Transformed token embeddings tensor of shape ``(B, N, C)``.

    Example:
        >>> encoder = Encoder(num_layers=12, embed_dim=768, num_heads=12, dropout=0.1, hidden_dim=3072)
        >>> x = torch.randn(2, 197, 768)
        >>> out = encoder(x)
        >>> out.shape
        torch.Size([2, 197, 768])
    """

    def __init__(
        self,
        num_layers: int = 12,
        embed_dim: int = 768,
        num_heads: int = 12,
        dropout: float = 0.1,
        hidden_dim: int = 3072,
        eps: float = 1e-5,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
        Emb_dim: int | None = None,
    ) -> None:
        super().__init__()
        if Emb_dim is not None:
            embed_dim = Emb_dim
        self.layers = nn.ModuleList([
            Block(embed_dim, num_heads, dropout, hidden_dim, eps, device=device, dtype=dtype)
            for _ in range(num_layers)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class ViT(nn.Module):
    """Vision Transformer for image classification.

    Simple explanation:
        ViT splits an image into fixed-size patches, projects each patch into an
        embedding vector, and processes the patch sequence with transformer
        encoder blocks. A learnable ``[CLS]`` token is prepended, and its final
        representation is used for classification.

    Architecture details (current implementation):
        - Task: image classification
        - Input tokenization: non-overlapping convolutional patch projection
        - Attention: multi-head self-attention (full global attention)
        - Masking: no causal mask (bidirectional attention over all patches)
        - Positional encoding: learnable absolute positional embeddings
        - Feed-forward block: GELU-based FFN (``FF_GELU``)
        - Normalization: Pre-Norm LayerNorm inside each transformer block
        - Head: LayerNorm + Linear classifier on ``[CLS]`` token

    Historical context:
        - Model family introduced by the paper "An Image is Worth 16x16 Words"
          (Google Research, 2020; published at ICLR 2021).
        - ViT showed that transformer-only backbones can match or exceed strong
          CNN baselines when trained with sufficient data.

    Paper reference:
        - ViT paper: https://arxiv.org/abs/2010.11929

    Example:
        >>> import torch
        >>> from stackformer.vision import ViT
        >>> model = ViT(img_size=224, patch_size=16, num_classes=1000)
        >>> x = torch.randn(2, 3, 224, 224)
        >>> logits = model(x)
        >>> logits.shape
        torch.Size([2, 1000])

    Args:
        img_size (int, default=224): Input image height/width (square expected).
        patch_size (int, default=16): Patch size used for tokenization.
        num_layers (int, default=12): Number of transformer encoder layers.
        embed_dim (int, default=768): Embedding dimension for patch tokens.
        num_classes (int, default=1000): Number of output classes.
        num_heads (int, default=12): Number of attention heads.
        dropout (float, default=0.1): Dropout probability.
        hidden_dim (int, default=3072): Hidden dimension of the feed-forward layer.
        eps (float, default=1e-5): Epsilon used by layer normalization.
        in_channels (int, default=3): Number of input image channels.
        Emb_dim (int | None, default=None): Deprecated alias for embed_dim.
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        num_layers: int = 12,
        embed_dim: int = 768,
        num_classes: int = 1000,
        num_heads: int = 12,
        dropout: float = 0.1,
        hidden_dim: int = 3072,
        eps: float = 1e-5,
        in_channels: int = 3,
        Emb_dim: int | None = None,
    ) -> None:
        super().__init__()
        if Emb_dim is not None:
            embed_dim = Emb_dim
        self.patch_embedding = PatchEmbedding(img_size, patch_size, embed_dim, in_channels)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + self.patch_embedding.num_patches, embed_dim))
        self.dropout = nn.Dropout(dropout)

        self.encoder = Encoder(num_layers, embed_dim, num_heads, dropout, hidden_dim, eps)

        # Classification head
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(embed_dim, eps=eps),
            nn.Linear(embed_dim, num_classes),
        )

        # Init weights
        self.apply(self._init_weights)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def _init_weights(self, m: nn.Module) -> None:
        """Initialize linear, conv, and layernorm layer weights."""
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0)
        x = self.patch_embedding(x)  # (B, N, C)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, C)
        pos_embed = self.pos_embed  # (1, 1+N, C)

        x = torch.cat((cls_tokens, x), dim=1)  # (B, 1+N, C)
        x = x + pos_embed
        x = self.dropout(x)

        x = self.encoder(x)  # (B, 1+N, C)
        return self.mlp_head(x[:, 0])  # (B, num_classes)

