import pytest
import torch
from tests._test_utils import _checkpoint

from stackformer.vision.vit import ViT
from stackformer.vision.segformer import (
    SegFormer,
    Encoder,
    patch,
    transformer_block,
    Multi_Head_Attention as SegFormerMHA,
)


def test_vit_forward(torch_device):
    _checkpoint("test_vit_forward setup", device=torch_device)
    B = 2
    img = torch.randn(B, 3, 32, 32, device=torch_device)

    model = ViT(
        img_size=32,
        patch_size=8,
        num_layers=1,
        Emb_dim=32,
        num_heads=4,
        hidden_dim=64,
        num_classes=10,
        dropout=0.0,
    ).to(torch_device)

    _checkpoint("Executing ViT forward pass")
    out = model(img)
    _checkpoint("Asserting ViT output shape", out_shape=out.shape)
    assert out.shape == (B, 10)
    assert torch.isfinite(out).all()


def test_patch_embedding(torch_device):
    _checkpoint("test_patch_embedding setup", device=torch_device)
    B = 2
    img = torch.randn(B, 3, 32, 32, device=torch_device)

    p = patch(patch_size=4, stride=4, in_channels=3, hidden_size=16).to(torch_device)
    _checkpoint("Executing patch embedding forward")
    out, h, w = p(img)

    # overlapping tokens = ((32 + 2*2 - 4)/4 + 1)^2 = 9^2 = 81
    _checkpoint("Asserting patch output shape", out_shape=out.shape)
    assert out.shape == (B, 81, 16)
    assert torch.isfinite(out).all()


def test_segformer_attention(torch_device):
    _checkpoint("test_segformer_attention setup (H-01 spatial reduction MHA)", device=torch_device)
    B, N, C = 2, 16, 32
    x = torch.randn(B, N, C, device=torch_device)

    att = SegFormerMHA(hidden_size=C, num_heads=4, sr_ratio=1).to(torch_device)
    _checkpoint("Executing SegFormer spatial reduction attention forward")
    out = att(x, h=4, w=4)

    _checkpoint("Asserting SegFormer attention output shape", out_shape=out.shape)
    assert out.shape == x.shape
    assert torch.isfinite(out).all()


def test_transformer_block(torch_device):
    _checkpoint("test_transformer_block setup", device=torch_device)
    B, N, C = 2, 16, 32
    x = torch.randn(B, N, C, device=torch_device)

    block = transformer_block(
        hidden_size=C,
        num_heads=4,
        sr_ratio=1,
        mlp_ratio=2,
        dropout=0.0,
    ).to(torch_device)

    _checkpoint("Executing SegFormer transformer block forward")
    out = block(x, h=4, w=4)
    _checkpoint("Asserting block output shape", out_shape=out.shape)
    assert out.shape == x.shape
    assert torch.isfinite(out).all()


def test_segformer_encoder(torch_device):
    _checkpoint("test_segformer_encoder setup", device=torch_device)
    B = 2
    img = torch.randn(B, 3, 64, 64, device=torch_device)

    enc = Encoder(variant="b0").to(torch_device)
    _checkpoint("Executing SegFormer Encoder forward")
    f1, f2, f3, f4 = enc(img)

    _checkpoint("Asserting multi-scale feature map dimensions")
    assert f1.ndim == 4
    assert f2.ndim == 4
    assert f3.ndim == 4
    assert f4.ndim == 4


def test_segformer_full_forward(torch_device):
    _checkpoint("test_segformer_full_forward setup", device=torch_device)
    B = 1
    img = torch.randn(B, 3, 64, 64, device=torch_device)

    model = SegFormer(num_labels=5).to(torch_device)
    _checkpoint("Executing SegFormerB0 full forward")
    out = model(img, upsample_to_input=True)

    _checkpoint("Asserting SegFormerB0 output shape", out_shape=out.shape)
    assert out.shape == (B, 5, 64, 64)
    assert torch.isfinite(out).all()


@pytest.mark.gpu
def test_segformer_gpu_device_assertion_regression():
    """Cover Critical C-01 issue: SegFormer device assertion crash on GPU.

    When SegFormerB0 is created on CPU and moved to GPU via .to('cuda'),
    local Multi_Head_Attention retains self.device = 'cpu', raising AssertionError.
    """
    _checkpoint("test_segformer_gpu_device_assertion_regression setup (C-01)")
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable for SegFormer GPU device assertion test")

    model = SegFormer(num_labels=5).to("cuda")
    img = torch.randn(1, 3, 64, 64, device="cuda")

    _checkpoint("Testing SegFormer GPU forward after .to('cuda')")
    out = model(img, upsample_to_input=True)
    assert out.shape == (1, 5, 64, 64)