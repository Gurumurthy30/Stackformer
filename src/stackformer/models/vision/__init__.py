"""Vision model architectures and building blocks.

Exposes:
    - ViT (vit): Vision Transformer architecture
    - SegFormer (segformer, SegFormerB0): Efficient transformer for semantic segmentation
"""

from .segformer import SegFormer
from .vit import ViT

# Common aliases
vit = ViT
segformer = SegFormer
SegFormerB0 = SegFormer

__all__ = [
    "SegFormer",
    "SegFormerB0",
    "ViT",
    "segformer",
    "vit",
]
