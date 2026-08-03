"""SegFormer-B0 architecture implementation for semantic segmentation.

Implements multi-scale hierarchical transformer encoder stages with Spatial Reduction Attention
and lightweight MLP decoder head.

Paper reference:
    SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers
    https://arxiv.org/abs/2105.15203
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from stackformer.modules.attention_engine import _run_sdpa
from stackformer.modules.Feed_forward import FF_GELU


class Patch(nn.Module):
    """Overlapping patch embedding layer for SegFormer encoder stages.

    Constructor args:
        img_size (int, default=224): Input spatial dimensions.
        in_channels (int, default=3): Input feature channels.
        out_channels (int, default=768): Output embedding channels.
        kernel (int, default=7): Convolution kernel size.
        stride (int, default=3): Convolution stride.
        padding (int, default=3): Convolution padding.

    Learnable parameters:
        proj.weight: Shape ``(out_channels, in_channels, kernel, kernel)``. Convolution weights.
        proj.bias: Shape ``(out_channels,)``. Convolution bias.

    Forward args:
        x (torch.Tensor): Input feature maps of shape ``(B, C, H, W)``.

    Returns:
        torch.Tensor: Flattened patch tokens of shape ``(B, N, C_out)``.

    Example:
        >>> p = Patch(img_size=224, in_channels=3, out_channels=32, kernel=7, stride=4, padding=3)
        >>> x = torch.randn(2, 3, 224, 224)
        >>> out = p(x)
        >>> out.shape
        torch.Size([2, 3136, 32])
    """

    def __init__(
        self,
        img_size: int = 224,
        in_channels: int = 3,
        out_channels: int = 768,
        kernel: int = 7,
        stride: int = 3,
        padding: int = 3,
    ) -> None:
        super().__init__()
        self.img_size = img_size
        self.kernel = kernel
        self.out_channels = out_channels
        self.num_patches = (img_size // kernel) ** 2

        self.proj = nn.Conv2d(
            in_channels=in_channels,
            out_channels=self.out_channels,
            kernel_size=kernel,
            stride=stride,
            padding=padding,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        x = self.proj(x)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # (B, H*W, C)
        return x


class SpatialReductionAttention(nn.Module):
    """Spatial Reduction Attention (SRA) for efficient multi-head self-attention.

    Downsamples Key and Value spatial sequences using 1D depthwise convolution to reduce
    quadratic computational complexity in high-resolution vision transformer stages.

    Constructor args:
        embed_dim (int): Embedding channels dimension.
        num_heads (int): Number of attention heads.
        dropout (float, default=0.0): Dropout probability.
        reduction (int, default=1): Spatial reduction ratio for Key/Value tokens.
        device (torch.device | str, default='cpu'): Target device.
        dtype (torch.dtype, default=torch.float32): Tensor data type.

    Learnable parameters:
        q_proj, k_proj, v_proj, out_proj: Linear projection layers.
        kv_down: Conv1d depthwise reduction module (if reduction > 1).

    Forward args:
        x (torch.Tensor): Input feature tensor of shape ``(B, N, C)``.

    Returns:
        torch.Tensor: Output features tensor of shape ``(B, N, C)``.

    Example:
        >>> sra = SpatialReductionAttention(embed_dim=32, num_heads=1, reduction=8)
        >>> x = torch.randn(2, 3136, 32)
        >>> out = sra(x)
        >>> out.shape
        torch.Size([2, 3136, 32])
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        reduction: int = 1,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.reduction = reduction

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=True, device=device, dtype=dtype)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=True, device=device, dtype=dtype)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=True, device=device, dtype=dtype)

        if reduction > 1:
            self.kv_down = nn.Conv1d(
                embed_dim, embed_dim, kernel_size=3, stride=reduction, padding=1, groups=embed_dim
            )
        else:
            self.kv_down = nn.Identity()

        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True, device=device, dtype=dtype)
        self.dropout_p = dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape

        q = self.q_proj(x)  # (B, N, C)

        if self.reduction > 1:
            x_down = x.transpose(1, 2)  # (B, C, N)
            x_down = self.kv_down(x_down).transpose(1, 2)  # (B, N_reduced, C)

            k = self.k_proj(x_down)  # (B, N_reduced, C)
            v = self.v_proj(x_down)  # (B, N_reduced, C)
        else:
            k = self.k_proj(x)  # (B, N, C)
            v = self.v_proj(x)  # (B, N, C)

        q = q.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, N, D)
        k = k.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, N_red, D)
        v = v.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, N_red, D)
        
        out = _run_sdpa(
                q, k, v,
                attn_mask=None,
                dropout_p=self.dropout_p
        )

        out = out.transpose(1, 2).contiguous().view(B, N, C)  # (B, N, C)

        return self.out_proj(out)


class TransformerBlock(nn.Module):
    """Single SegFormer transformer block with Spatial Reduction Attention and Mix-FFN.

    Constructor args:
        embed_dim (int): Embedding dimension.
        num_heads (int): Number of attention heads.
        hidden_dim (int): Inner hidden dimension of feed-forward network.
        dropout (float): Dropout probability.
        reduction (int): Spatial reduction ratio for SRA.

    Forward args:
        x (torch.Tensor): Input token features of shape ``(B, N, C)``.

    Returns:
        torch.Tensor: Transformed token features of shape ``(B, N, C)``.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        hidden_dim: int,
        dropout: float,
        reduction: int,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.att = SpatialReductionAttention(
            embed_dim=embed_dim, num_heads=num_heads, dropout=dropout, reduction=reduction
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ff = FF_GELU(embed_dim=embed_dim, hidden_dim=hidden_dim, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.att(self.norm1(x))
        x = x + self.ff(self.norm2(x))
        return x


class TransformerEncoder(nn.Module):
    """Stack of SegFormer transformer blocks for a single stage.

    Constructor args:
        num_layers (int): Number of transformer blocks in this stage.
        embed_dim (int): Embedding dimension for this stage.
        num_heads (int): Number of attention heads.
        hidden_dim (int): Inner hidden dimension of FFN.
        dropout (float): Dropout probability.
        reduction (int): Spatial reduction ratio.

    Forward args:
        x (torch.Tensor): Input stage tokens tensor of shape ``(B, N, C)``.

    Returns:
        torch.Tensor: Output stage tokens tensor of shape ``(B, N, C)``.
    """

    def __init__(
        self,
        num_layers: int,
        embed_dim: int,
        num_heads: int,
        hidden_dim: int,
        dropout: float,
        reduction: int,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, hidden_dim, dropout, reduction)
            for _ in range(num_layers)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class Encoder(nn.Module):
    """Hierarchical 4-stage SegFormer-B0 encoder.

    Extracts multi-scale feature maps at resolutions 1/4, 1/8, 1/16, and 1/32 of input size.

    Constructor args:
        None (instantiates standard SegFormer-B0 stage parameters).

    Forward args:
        x (torch.Tensor): Input image tensor of shape ``(B, 3, 224, 224)``.

    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: 4 feature maps of shapes
        ``(B, 32, 56, 56)``, ``(B, 64, 28, 28)``, ``(B, 160, 14, 14)``, ``(B, 256, 7, 7)``.
    """

    def __init__(self) -> None:
        super().__init__()

        # SegFormer B0 stage 1 configuration
        self.s1 = Patch(img_size=224, in_channels=3, out_channels=32, kernel=7, stride=4, padding=3)
        self.te1 = TransformerEncoder(
            num_layers=2, embed_dim=32, num_heads=1, hidden_dim=32 * 8, dropout=0.0, reduction=8
        )

        # Stage 2
        self.s2 = Patch(img_size=56, in_channels=32, out_channels=64, kernel=3, stride=2, padding=1)
        self.te2 = TransformerEncoder(
            num_layers=2, embed_dim=64, num_heads=2, hidden_dim=64 * 4, dropout=0.0, reduction=4
        )

        # Stage 3
        self.s3 = Patch(img_size=28, in_channels=64, out_channels=160, kernel=3, stride=2, padding=1)
        self.te3 = TransformerEncoder(
            num_layers=2, embed_dim=160, num_heads=5, hidden_dim=160 * 4, dropout=0.0, reduction=2
        )

        # Stage 4
        self.s4 = Patch(img_size=14, in_channels=160, out_channels=256, kernel=3, stride=2, padding=1)
        self.te4 = TransformerEncoder(
            num_layers=2, embed_dim=256, num_heads=8, hidden_dim=256 * 4, dropout=0.0, reduction=1
        )

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

        # ---------------- Stage 1 ----------------
        x1 = self.s1(x)  # (B, N1, 32)
        f1 = self.te1(x1)

        B, N1, C1 = f1.shape
        H1 = W1 = int(N1**0.5)
        f1_spatial = f1.transpose(1, 2).reshape(B, C1, H1, W1)  # (B, 32, H1, W1)

        # ---------------- Stage 2 ----------------
        x2 = self.s2(f1_spatial)  # (B, N2, 64)
        f2 = self.te2(x2)

        B, N2, C2 = f2.shape
        H2 = W2 = int(N2**0.5)
        f2_spatial = f2.transpose(1, 2).reshape(B, C2, H2, W2)  # (B, 64, H2, W2)

        # ---------------- Stage 3 ----------------
        x3 = self.s3(f2_spatial)  # (B, N3, 160)
        f3 = self.te3(x3)

        B, N3, C3 = f3.shape
        H3 = W3 = int(N3**0.5)
        f3_spatial = f3.transpose(1, 2).reshape(B, C3, H3, W3)  # (B, 160, H3, W3)

        # ---------------- Stage 4 ----------------
        x4 = self.s4(f3_spatial)  # (B, N4, 256)
        f4 = self.te4(x4)

        B, N4, C4 = f4.shape
        H4 = W4 = int(N4**0.5)
        f4_spatial = f4.transpose(1, 2).reshape(B, C4, H4, W4)  # (B, 256, H4, W4)

        return f1_spatial, f2_spatial, f3_spatial, f4_spatial


class MLP(nn.Module):
    """Lightweight All-MLP Decoder head for SegFormer feature fusion.

    Constructor args:
        out_dim (int, default=150): Number of output segmentation classes.

    Forward args:
        features (tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]): 4-stage encoder feature maps.

    Returns:
        torch.Tensor: Fused segmentation logits tensor of shape ``(B, out_dim, H1, W1)``.
    """

    def __init__(self, out_dim: int = 150) -> None:
        super().__init__()

        self.f1_proj = nn.Linear(32, 256)
        self.f2_proj = nn.Linear(64, 256)
        self.f3_proj = nn.Linear(160, 256)
        self.f4_proj = nn.Linear(256, 256)

        self.classifier = nn.Linear(256 * 4, out_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(
        self, features: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        f1, f2, f3, f4 = features
        B = f1.shape[0]

        target_size = f1.shape[2:]  # (H1, W1)

        # Bilinear upsampling to target spatial size
        f2_up = F.interpolate(f2, size=target_size, mode="bilinear", align_corners=False)
        f3_up = F.interpolate(f3, size=target_size, mode="bilinear", align_corners=False)
        f4_up = F.interpolate(f4, size=target_size, mode="bilinear", align_corners=False)

        # Flatten features for linear projection
        f1_flat = f1.flatten(2).transpose(1, 2)  # (B, H*W, 32)
        f2_flat = f2_up.flatten(2).transpose(1, 2)  # (B, H*W, 64)
        f3_flat = f3_up.flatten(2).transpose(1, 2)  # (B, H*W, 160)
        f4_flat = f4_up.flatten(2).transpose(1, 2)  # (B, H*W, 256)

        f1_proj = self.f1_proj(f1_flat)  # (B, H*W, 256)
        f2_proj = self.f2_proj(f2_flat)  # (B, H*W, 256)
        f3_proj = self.f3_proj(f3_flat)  # (B, H*W, 256)
        f4_proj = self.f4_proj(f4_flat)  # (B, H*W, 256)

        fused = torch.cat([f1_proj, f2_proj, f3_proj, f4_proj], dim=-1)  # (B, H*W, 1024)

        fused = self.dropout(fused)
        output = self.classifier(fused)  # (B, H*W, out_dim)

        H, W = target_size
        output = output.transpose(1, 2).reshape(B, -1, H, W)  # (B, out_dim, H, W)

        return output


class SegFormerB0(nn.Module):
    """SegFormer-B0 style model for semantic segmentation.

    Simple explanation:
        SegFormer takes an input image and builds feature maps at multiple
        scales (from high resolution to low resolution) using hierarchical
        transformer stages. These features are fused by an MLP decoder to
        produce per-pixel class predictions.

    Architecture details (current implementation):
        - Task: semantic segmentation
        - Encoder type: hierarchical transformer with 4 stages
        - Stage channels: [32, 64, 160, 256]
        - Attention: efficient multi-head self-attention with sequence
          reduction factors [8, 4, 2, 1] across stages (``SpatialReductionAttention``)
        - Masking: no causal mask (full bidirectional self-attention)
        - Positional strategy: implicit positional cues from overlapping patch
          embeddings (no explicit absolute position embedding tensor)
        - Feed-forward block: GELU MLP per transformer block
        - Normalization: Pre-Norm LayerNorm in each transformer block
        - Decoder: MLP feature projection + multi-scale fusion + bilinear upsampling

    Historical context:
        - SegFormer was introduced by NVIDIA in 2021 as a simple and effective
          transformer architecture for semantic segmentation.
        - It avoids heavy decoder designs and explicit positional encodings
          while keeping strong accuracy-speed trade-offs.

    Paper reference:
        - SegFormer paper: https://arxiv.org/abs/2105.15203

    Example:
        >>> import torch
        >>> from stackformer.vision import SegFormerB0
        >>> model = SegFormerB0(num_classes=19)
        >>> x = torch.randn(2, 3, 224, 224)
        >>> y = model(x)
        >>> y.shape
        torch.Size([2, 19, 224, 224])

    Args:
        num_classes (int, default=150): Number of segmentation classes in the output map.
    """

    def __init__(self, num_classes: int = 150) -> None:
        super().__init__()
        self.encoder = Encoder()
        self.decoder = MLP(out_dim=num_classes)

        # Final projection layer: upsample to original input size
        self.final_upsample = nn.Upsample(scale_factor=4, mode="bilinear", align_corners=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        H, W = x.shape[2:]
        features = self.encoder(x)
        output = self.decoder(features)  # (B, num_classes, H1, W1)

        output = F.interpolate(output, size=(H, W), mode="bilinear", align_corners=False)
        return output


# Backward compatibility aliases for internal test imports
patch = Patch
Multi_Head_Attention = SpatialReductionAttention
transformer_block = TransformerBlock
transformer_encoder = TransformerEncoder
