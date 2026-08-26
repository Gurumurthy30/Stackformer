"""SegFormer / MixVisionTransformer architecture implementation for semantic segmentation.

Provides hierarchical multi-stage vision encoders with spatial reduction attention (SRA) and Mix-FFN blocks,
along with all-MLP decode heads for semantic segmentation tasks.

Paper reference:
    SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers
    https://arxiv.org/abs/2105.15203
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from stackformer.modules.attention_engine import _run_sdpa
except ImportError:

    def _run_sdpa(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        dropout_p: float = 0.0,
        is_causal: bool = False,
    ) -> torch.Tensor:
        return F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal
        )


MIT_CONFIGS = {
    "b0": dict(embed_dims=[32, 64, 160, 256], depths=[2, 2, 2, 2], decoder_hidden_size=256),
    "b1": dict(embed_dims=[64, 128, 320, 512], depths=[2, 2, 2, 2], decoder_hidden_size=768),
    "b2": dict(embed_dims=[64, 128, 320, 512], depths=[3, 4, 6, 3], decoder_hidden_size=768),
    "b3": dict(embed_dims=[64, 128, 320, 512], depths=[3, 4, 18, 3], decoder_hidden_size=768),
    "b4": dict(embed_dims=[64, 128, 320, 512], depths=[3, 8, 27, 3], decoder_hidden_size=768),
    "b5": dict(embed_dims=[64, 128, 320, 512], depths=[3, 6, 40, 3], decoder_hidden_size=768),
}

NUM_HEADS = [1, 2, 5, 8]
SR_RATIOS = [8, 4, 2, 1]
PATCH_SIZES = [7, 3, 3, 3]
STRIDES = [4, 2, 2, 2]
MLP_RATIO = 4


class OverlapPatchEmbeddings(nn.Module):
    """Strided overlapping convolutional patch embedding layer for SegFormer.

    Constructor args:
        patch_size (int): Size of 2D kernel.
        stride (int): Convolutional stride.
        in_channels (int): Input feature channels.
        hidden_size (int): Output embedding dimension.

    Forward args:
        pixel_values (torch.Tensor): Input image tensor of shape ``(B, C, H, W)``.

    Returns:
        tuple[torch.Tensor, int, int]: Tuple of patch tokens tensor ``(B, N, C)`` and spatial dimensions (H, W).
    """

    def __init__(self, patch_size: int, stride: int, in_channels: int, hidden_size: int) -> None:
        super().__init__()
        self.proj = nn.Conv2d(
            in_channels, hidden_size, kernel_size=patch_size, stride=stride, padding=patch_size // 2
        )
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, pixel_values: torch.Tensor) -> tuple[torch.Tensor, int, int]:
        x = self.proj(pixel_values)
        _, _, h, w = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.layer_norm(x)
        return x, h, w


class SequenceReduction(nn.Module):
    """Strided Conv2d + LayerNorm used to downsample Key/Value tokens before attention."""

    def __init__(self, hidden_size: int, sr_ratio: int) -> None:
        super().__init__()
        self.sequence_reduction = nn.Conv2d(hidden_size, hidden_size, kernel_size=sr_ratio, stride=sr_ratio)
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, x: torch.Tensor, h: int, w: int) -> torch.Tensor:
        b, n, c = x.shape
        x = x.transpose(1, 2).reshape(b, c, h, w)
        x = self.sequence_reduction(x)
        x = x.reshape(b, c, -1).transpose(1, 2)
        x = self.layer_norm(x)
        return x


class EfficientAttention(nn.Module):
    """Multi-head self-attention with spatially-reduced Key/Value tokens (SRA)."""

    def __init__(self, hidden_size: int, num_heads: int, sr_ratio: int) -> None:
        super().__init__()
        assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.sr_ratio = sr_ratio

        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.o_proj = nn.Linear(hidden_size, hidden_size)

        if sr_ratio > 1:
            self.sequence_reduction = SequenceReduction(hidden_size, sr_ratio)

    def forward(self, x: torch.Tensor, h: int, w: int) -> torch.Tensor:
        b, n, c = x.shape
        q = self.q_proj(x).view(b, n, self.num_heads, self.head_dim).transpose(1, 2)

        kv_in = self.sequence_reduction(x, h, w) if self.sr_ratio > 1 else x
        n_kv = kv_in.shape[1]
        k = self.k_proj(kv_in).view(b, n_kv, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(kv_in).view(b, n_kv, self.num_heads, self.head_dim).transpose(1, 2)

        out = _run_sdpa(q, k, v, attn_mask=None, dropout_p=0.0)
        out = out.transpose(1, 2).reshape(b, n, c)
        return self.o_proj(out)


class DepthWiseConv(nn.Module):
    """3x3 depthwise conv that injects positional information into Mix-FFN."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim)

    def forward(self, x: torch.Tensor, h: int, w: int) -> torch.Tensor:
        b, n, c = x.shape
        x = x.transpose(1, 2).view(b, c, h, w)
        x = self.dwconv(x)
        return x.flatten(2).transpose(1, 2)


class MixFFN(nn.Module):
    """Mix-FFN module: fc1 -> depthwise 3x3 conv -> GELU -> fc2."""

    def __init__(self, in_features: int, hidden_features: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DepthWiseConv(hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, h: int, w: int) -> torch.Tensor:
        x = self.fc1(x)
        x = self.dwconv(x, h, w)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return self.dropout(x)


class Block(nn.Module):
    """Pre-norm SegFormer encoder block with Spatial Reduction Attention (SRA) and Mix-FFN."""

    def __init__(
        self, hidden_size: int, num_heads: int, sr_ratio: int, mlp_ratio: int, dropout: float = 0.0
    ) -> None:
        super().__init__()
        self.layernorm_before = nn.LayerNorm(hidden_size)
        self.attention = EfficientAttention(hidden_size, num_heads, sr_ratio)
        self.layernorm_after = nn.LayerNorm(hidden_size)
        self.mlp = MixFFN(hidden_size, hidden_size * mlp_ratio, dropout=dropout)

    def forward(self, x: torch.Tensor, h: int, w: int) -> torch.Tensor:
        x = x + self.attention(self.layernorm_before(x), h, w)
        x = x + self.mlp(self.layernorm_after(x), h, w)
        return x


class Stage(nn.Module):
    """One SegFormer encoder stage: overlap patch embedding -> N blocks -> final LayerNorm."""

    def __init__(
        self,
        in_channels: int,
        hidden_size: int,
        depth: int,
        num_heads: int,
        sr_ratio: int,
        patch_size: int,
        stride: int,
        mlp_ratio: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.patch_embeddings = OverlapPatchEmbeddings(patch_size, stride, in_channels, hidden_size)
        self.blocks = nn.ModuleList(
            [Block(hidden_size, num_heads, sr_ratio, mlp_ratio, dropout) for _ in range(depth)]
        )
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, h, w = self.patch_embeddings(x)
        for block in self.blocks:
            x = block(x, h, w)
        x = self.layer_norm(x)
        b = x.shape[0]
        return x.reshape(b, h, w, -1).permute(0, 3, 1, 2).contiguous()


class MixVisionEncoder(nn.Module):
    """4-stage hierarchical Mix Vision Transformer (MiT) encoder."""

    def __init__(self, variant: str, dropout: float = 0.0) -> None:
        super().__init__()
        cfg = MIT_CONFIGS[variant]
        embed_dims, depths = cfg["embed_dims"], cfg["depths"]
        stages = []
        for i in range(4):
            in_ch = 3 if i == 0 else embed_dims[i - 1]
            stages.append(
                Stage(
                    in_channels=in_ch,
                    hidden_size=embed_dims[i],
                    depth=depths[i],
                    num_heads=NUM_HEADS[i],
                    sr_ratio=SR_RATIOS[i],
                    patch_size=PATCH_SIZES[i],
                    stride=STRIDES[i],
                    mlp_ratio=MLP_RATIO,
                    dropout=dropout,
                )
            )
        self.stages = nn.ModuleList(stages)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        outs = []
        for stage in self.stages:
            x = stage(x)
            outs.append(x)
        return outs


class MLPProj(nn.Module):
    """Per-stage linear projection to decoder_hidden_size (HF's ``SegformerMLP``)."""

    def __init__(self, in_dim: int, decoder_hidden_size: int) -> None:
        super().__init__()
        self.proj = nn.Linear(in_dim, decoder_hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x.flatten(2).transpose(1, 2))


class DecodeHead(nn.Module):
    """All-MLP decode head: per-stage linear proj -> upsample -> concat -> conv fuse -> classifier."""

    def __init__(self, variant: str, num_labels: int, dropout: float = 0.1) -> None:
        super().__init__()
        cfg = MIT_CONFIGS[variant]
        embed_dims, dh = cfg["embed_dims"], cfg["decoder_hidden_size"]
        self.linear_projections = nn.ModuleList([MLPProj(d, dh) for d in embed_dims])
        self.linear_fuse = nn.Conv2d(dh * 4, dh, kernel_size=1, bias=False)
        self.batch_norm = nn.BatchNorm2d(dh)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Conv2d(dh, num_labels, kernel_size=1)

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        b = features[0].shape[0]
        target_hw = features[0].shape[2:]
        projected = []
        for feat, proj in zip(features, self.linear_projections):
            h, w = feat.shape[2:]
            t = proj(feat).transpose(1, 2).reshape(b, -1, h, w)
            t = F.interpolate(t, size=target_hw, mode="bilinear", align_corners=False)
            projected.append(t)

        fused = self.linear_fuse(torch.cat(projected[::-1], dim=1))
        fused = self.batch_norm(fused)
        fused = self.activation(fused)
        fused = self.dropout(fused)
        return self.classifier(fused)


class StackFormerSegformer(nn.Module):
    """Full SegFormer semantic segmentation model (MixVisionEncoder + DecodeHead).

    Simple explanation:
        SegFormer processes input images through a 4-stage hierarchical Mix Vision Transformer
        encoder with Spatial Reduction Attention (SRA) and Mix-FFN blocks, followed by an
        all-MLP decode head that fuses multi-scale features for pixel-level classification.

    Architecture details (current implementation):
        - Task: semantic segmentation
        - Attention: Spatial Reduction Attention (SRA) with dynamic sequence reduction
        - Masking: none (bidirectional 2D spatial attention)
        - Positional encoding: zero explicit positional embeddings (position encoded via Mix-FFN 3x3 depthwise convs)
        - Feed-forward: Mix-FFN (FC -> 3x3 DWConv -> GELU -> FC)
        - Normalization: Pre-Norm LayerNorm in encoder; BatchNorm2d in decode head
        - Head: All-MLP decode head with 1x1 conv fusion and linear classifier

    Historical context:
        - Introduced by Xie et al. in 2021 ("SegFormer: Simple and Efficient Design for Semantic Segmentation").
        - Demonstrated state-of-the-art segmentation efficiency by avoiding positional embeddings and complex decoders.

    Paper reference:
        - SegFormer paper: https://arxiv.org/abs/2105.15203

    Example:
        >>> import torch
        >>> from stackformer.vision import SegFormerB0
        >>> model = SegFormerB0(num_labels=150)
        >>> x = torch.randn(2, 3, 512, 512)
        >>> logits = model(x)
        >>> logits.shape
        torch.Size([2, 150, 128, 128])

    Args:
        variant (str, default="b4"): SegFormer model variant ("b0", "b1", "b2", "b3", "b4", "b5").
        num_labels (int, default=150): Number of semantic segmentation target classes.
        dropout (float, default=0.0): Encoder dropout probability.
    """

    def __init__(self, variant: str = "b4", num_labels: int = 150, dropout: float = 0.0) -> None:
        super().__init__()
        self.segformer = MixVisionEncoder(variant, dropout=dropout)
        self.decode_head = DecodeHead(variant, num_labels, dropout=0.1)

    def forward(self, pixel_values: torch.Tensor, upsample_to_input: bool = False) -> torch.Tensor:
        features = self.segformer(pixel_values)
        logits = self.decode_head(features)
        if upsample_to_input:
            logits = F.interpolate(
                logits, size=pixel_values.shape[2:], mode="bilinear", align_corners=False
            )
        return logits


class SegFormer(StackFormerSegformer):
    """SegFormer variant preset container."""
    
    def __init__(self, variant: str = "b0", num_labels: int = 150, dropout: float = 0.0) -> None:
        super().__init__(variant=variant, num_labels=num_labels, dropout=dropout)


# PascalCase renames per Section 5 of style guide
Patch = OverlapPatchEmbeddings
TransformerBlock = Block
TransformerEncoder = MixVisionEncoder

# Lowercase / snake_case aliases for backward compatibility
patch = Patch
transformer_block = TransformerBlock
transformer_encoder = TransformerEncoder