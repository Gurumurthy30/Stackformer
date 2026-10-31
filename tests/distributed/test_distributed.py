import os
import torch
from tests._test_utils import _checkpoint

from stackformer.distributed import barrier, cleanup_distributed, init_distributed, is_main_process, wrap_model_ddp


def test_distributed_gloo_single_process_init():
    _checkpoint("test_distributed_gloo_single_process_init setting environment variables")
    os.environ["RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"
    os.environ["LOCAL_RANK"] = "0"

    _checkpoint("Calling init_distributed")
    initialized = init_distributed(backend="gloo")
    _checkpoint("Asserting single process returns False for distributed init", initialized=initialized)
    assert initialized is False
    assert is_main_process()
    barrier()
    cleanup_distributed()
    assert torch.distributed.is_initialized() is False


def test_wrap_model_ddp_uninitialized_returns_raw_model(torch_device):
    """Cover H-02 issue verifying wrap_model_ddp when distributed is uninitialized."""
    _checkpoint("test_wrap_model_ddp_uninitialized_returns_raw_model setup", device=torch_device)
    model = torch.nn.Linear(8, 4, device=torch_device)
    wrapped = wrap_model_ddp(model)
    _checkpoint("Asserting wrap_model_ddp returns raw model when not distributed")
    assert wrapped is model
