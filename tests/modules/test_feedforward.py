import torch

from stackformer.modules.feed_forward import (
    FF_GELU,
    FF_GeGLU,
    FF_LeakyReLU,
    FF_ReLU,
    FF_Sigmoid,
    FF_SiLU,
    FF_SwiGLU,
)
from tests._test_utils import _checkpoint


def test_feedforward_variants_shape(torch_device):
    _checkpoint("test_feedforward_variants_shape setup", device=torch_device)
    x = torch.randn(2, 4, 8, device=torch_device)

    for ff in (
        FF_ReLU(8, 16, dropout=0.0, device=torch_device),
        FF_LeakyReLU(8, 16, dropout=0.0, device=torch_device),
        FF_GELU(8, 16, dropout=0.0, device=torch_device),
        FF_Sigmoid(8, 16, dropout=0.0, device=torch_device),
        FF_SiLU(8, 16, dropout=0.0, device=torch_device),
        FF_SwiGLU(8, 16, dropout=0.0, device=torch_device),
        FF_GeGLU(8, 16, dropout=0.0, device=torch_device),
    ):
        _checkpoint("Testing FFN module forward", class_name=ff.__class__.__name__)
        out = ff(x)
        assert out.shape == x.shape
        assert torch.isfinite(out).all()


def test_gated_mlp_parameter_counts(torch_device):
    """Cover M-02 issue verifying parameter count structure of gated vs non-gated MLPs."""
    _checkpoint("test_gated_mlp_parameter_counts setup", device=torch_device)
    ff_gelu = FF_GELU(8, 16, dropout=0.0, device=torch_device)
    ff_swiglu = FF_SwiGLU(8, 16, dropout=0.0, device=torch_device)

    gelu_params = sum(p.numel() for p in ff_gelu.parameters())
    swiglu_params = sum(p.numel() for p in ff_swiglu.parameters())

    _checkpoint("Comparing parameter counts", gelu_params=gelu_params, swiglu_params=swiglu_params)
    assert swiglu_params > gelu_params
