import torch

from stackformer.modules.normalization import LayerNorm as LayerNormalization
from stackformer.modules.normalization import RMSNorm as RMSNormalization
from tests._test_utils import _checkpoint


def test_normalization_outputs_are_finite(torch_device):
    _checkpoint("test_normalization_outputs_are_finite setup", device=torch_device)
    x = torch.randn(2, 4, 8, device=torch_device)
    _checkpoint("Checking LayerNormalization and RMSNormalization output finiteness")
    assert torch.isfinite(LayerNormalization(embed_dim=8, device=torch_device)(x)).all()
    assert torch.isfinite(RMSNormalization(embed_dim=8, device=torch_device)(x)).all()


def test_normalization_zero_input_stable(torch_device):
    _checkpoint("test_normalization_zero_input_stable setup", device=torch_device)
    x = torch.zeros(2, 4, 8, device=torch_device)
    _checkpoint("Checking zero input stability")
    assert torch.isfinite(LayerNormalization(embed_dim=8, device=torch_device)(x)).all()
    assert torch.isfinite(RMSNormalization(embed_dim=8, device=torch_device)(x)).all()


def test_layernorm_matches_torch_functional_reference(torch_device):
    """Finding #47: LayerNorm matches torch.nn.functional.layer_norm to high numerical precision."""
    import torch.nn as nn
    import torch.nn.functional as F

    embed_dim = 16
    x = torch.randn(3, 7, embed_dim, device=torch_device)

    ln = LayerNormalization(embed_dim=embed_dim, device=torch_device)
    nn.init.normal_(ln.weight)
    nn.init.normal_(ln.bias)

    actual = ln(x)
    expected = F.layer_norm(x, (embed_dim,), ln.weight, ln.bias, eps=ln.eps)
    assert torch.allclose(actual, expected, atol=1e-5), (
        "LayerNorm must match torch.nn.functional.layer_norm"
    )


def test_rmsnorm_matches_fp64_manual_reference(torch_device):
    """Finding #47: RMSNorm matches double-precision manual formula reference."""
    import torch.nn as nn

    embed_dim = 16
    x = torch.randn(3, 7, embed_dim, device=torch_device)
    rms = RMSNormalization(embed_dim=embed_dim, device=torch_device)
    nn.init.normal_(rms.weight)

    actual = rms(x)

    # Manual fp64 computation
    x_fp64 = x.to(torch.float64)
    w_fp64 = rms.weight.to(torch.float64)
    variance = x_fp64.pow(2).mean(dim=-1, keepdim=True)
    rsqrt = torch.rsqrt(variance + rms.eps)
    expected = (x_fp64 * rsqrt * w_fp64).to(x.dtype)

    assert torch.allclose(actual, expected, atol=1e-5), "RMSNorm must match fp64 reference formula"
