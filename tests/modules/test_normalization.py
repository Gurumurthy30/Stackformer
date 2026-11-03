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
