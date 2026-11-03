import pytest
import torch

from tests._test_utils import _checkpoint


@pytest.mark.gpu
def test_cuda_tensor_addition() -> None:
    _checkpoint("test_cuda_tensor_addition checking CUDA availability")
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")

    _checkpoint("Allocating CUDA tensors")
    a = torch.ones(8, device="cuda")
    b = torch.full((8,), 2.0, device="cuda")
    result = a + b

    _checkpoint("Asserting CUDA tensor sum", result_device=result.device, shape=result.shape)
    assert result.is_cuda
    assert torch.allclose(result.cpu(), torch.full((8,), 3.0))
