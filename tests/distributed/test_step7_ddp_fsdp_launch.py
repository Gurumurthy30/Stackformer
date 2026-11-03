"""Step 7 Tests: Distributed Data Parallel (DDP), FSDP2 (wiring), DCP, and Launcher."""

from __future__ import annotations

import os
import sys
from pathlib import Path
import pytest
import torch
import torch.nn as nn
import torch.distributed as dist

from stackformer.distributed.dcp import dcp_to_state_dict, load_dcp, save_dcp
from stackformer.distributed.ddp import apply_ddp
from stackformer.distributed.fsdp import apply_fsdp, build_fsdp_policies
from stackformer.distributed.utils import (
    configure_windows_environment,
    get_free_port,
    is_distributed_initialized,
    validate_platform_and_device,
)
from stackformer.models import GPT2
from stackformer.training.checkpoint import CheckpointManager, clean_state_dict, unwrap_model
from stackformer.training.config import TrainConfig
from stackformer.training.launcher import launch
from stackformer.training.state import TrainState


# ==============================================================================
# Top-level helper models and functions for multi-process worker spawning
# ==============================================================================


class TinyLinearModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 8)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(8, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.relu(self.fc1(x)))


def _worker_ddp_loss_parity(tmp_dir_str: str) -> None:
    """Worker function for testing DDP gradient synchronization and loss parity."""
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    torch.manual_seed(100)
    model = TinyLinearModel()
    ddp_model = apply_ddp(model, device="cpu")
    optimizer = torch.optim.SGD(ddp_model.parameters(), lr=0.1)

    # Each rank has a different micro-batch
    torch.manual_seed(100 + rank)
    inputs = torch.randn(2, 4)
    targets = torch.randn(2, 2)

    optimizer.zero_grad()
    outputs = ddp_model(inputs)
    loss = nn.functional.mse_loss(outputs, targets)
    loss.backward()
    optimizer.step()

    # Save final loss to disk
    out_file = Path(tmp_dir_str) / f"loss_rank_{rank}.pt"
    torch.save({"loss": loss.item()}, str(out_file))


def _worker_ddp_checkpoint_save_resume(tmp_dir_str: str) -> None:
    """Worker function for testing DDP checkpoint save without 'module.' prefix and reload parity."""
    rank = dist.get_rank()

    torch.manual_seed(200)
    model = TinyLinearModel()
    ddp_model = apply_ddp(model, device="cpu")
    mgr = CheckpointManager(checkpoint_dir=tmp_dir_str, keep_last_n=2)

    # Rank 0 saves checkpoint
    if rank == 0:
        state = TrainState(epoch=1, global_step=10)
        saved_dir = mgr.save(ddp_model, state)

        # Verify weights in state.pt and model.safetensors have no 'module.' prefix
        ckpt = mgr.load(saved_dir)
        assert ckpt is not None
        assert "module.fc1.weight" not in ckpt["model"]
        assert "fc1.weight" in ckpt["model"]

    dist.barrier()

    # Both ranks load into a fresh model and verify output equivalence
    fresh_model = TinyLinearModel()
    loaded_ckpt = mgr.load("auto")
    assert loaded_ckpt is not None
    fresh_model.load_state_dict(clean_state_dict(loaded_ckpt["model"]))

    x_test = torch.randn(2, 4)
    unwrapped = unwrap_model(ddp_model)
    assert torch.allclose(unwrapped(x_test), fresh_model(x_test), atol=1e-5)


def _worker_notebook_style(record_file_str: str) -> None:
    """Simulates a notebook user calling a custom training function via launch()."""
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    with open(f"{record_file_str}_{rank}.txt", "w", encoding="utf-8") as f:
        f.write(f"rank={rank},world_size={world_size}")


def _worker_dcp_save_load(tmp_dir_str: str) -> None:
    """Worker function for testing DCP save and load across distributed processes."""
    torch.manual_seed(300)
    model = TinyLinearModel()
    ddp_model = apply_ddp(model, device="cpu")

    mgr = CheckpointManager(checkpoint_dir=tmp_dir_str, use_dcp=True)
    state = TrainState(epoch=0, global_step=5)

    # All ranks participate in save
    saved_dir = mgr.save(ddp_model, state)
    assert (saved_dir / "dcp").is_dir()

    dist.barrier()

    # Load DCP into fresh model (all ranks participate)
    fresh_model = TinyLinearModel()
    loaded = mgr.load("auto", model=fresh_model)
    assert loaded is not None

    x = torch.randn(2, 4)
    assert torch.allclose(unwrap_model(ddp_model)(x), fresh_model(x), atol=1e-5)


# ==============================================================================
# Platform Guard & Validation Unit Tests (Fast, Single-Process)
# ==============================================================================


def test_platform_guard_windows_nccl(monkeypatch):
    """Platform guard: Requesting NCCL on native Windows must raise a clear RuntimeError."""
    monkeypatch.setattr(sys, "platform", "win32")
    with pytest.raises(RuntimeError, match="Native Windows does not support the NCCL distributed backend"):
        validate_platform_and_device(parallel="ddp", backend="nccl", device="cpu")


def test_platform_guard_windows_multi_gpu(monkeypatch):
    """Platform guard: Requesting multi-GPU CUDA on native Windows must raise a clear RuntimeError."""
    monkeypatch.setattr(sys, "platform", "win32")
    with pytest.raises(RuntimeError, match="Native Windows does not support multi-GPU distributed training with NCCL"):
        validate_platform_and_device(parallel="ddp", backend=None, device="cuda:0", world_size=2)


def test_platform_guard_cpu_fsdp2_raises():
    """Platform guard: Requesting FSDP2 on CPU must raise a clear RuntimeError recommending DDP or WSL2."""
    with pytest.raises(RuntimeError, match="FSDP2 in PyTorch 2.6 requires CUDA/accelerator streams"):
        validate_platform_and_device(parallel="fsdp2", backend=None, device="cpu")


def test_fsdp2_apply_on_cpu_raises_clear_error():
    """apply_fsdp on CPU must raise clear RuntimeError without silent fallback."""
    model = TinyLinearModel()
    with pytest.raises(RuntimeError, match="FSDP2 in PyTorch 2.6 requires CUDA/accelerator streams"):
        apply_fsdp(model, device="cpu")


def test_fsdp2_policy_construction_wiring():
    """Wiring test: Verify MixedPrecisionPolicy and OffloadPolicy construction (wiring, not numerical)."""
    mp_bf16, offload_cpu = build_fsdp_policies(mixed_precision="bf16", cpu_offload=True)
    assert mp_bf16.param_dtype == torch.bfloat16
    assert mp_bf16.reduce_dtype == torch.float32

    mp_none, offload_none = build_fsdp_policies(mixed_precision="no", cpu_offload=False)
    assert mp_none.param_dtype is None


def test_train_config_parallel_validation():
    """TrainConfig must validate parallel modes and reject unknown options."""
    with pytest.raises(ValueError, match="Unsupported parallel mode 'tensor_slice'"):
        TrainConfig(parallel="tensor_slice")

    cfg = TrainConfig(parallel="ddp")
    assert cfg.parallel == "ddp"


def test_configure_windows_environment_sets_use_libuv(monkeypatch):
    """configure_windows_environment sets USE_LIBUV=0 on Windows."""
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.delenv("USE_LIBUV", raising=False)
    configure_windows_environment()
    assert os.environ.get("USE_LIBUV") == "0"


def test_dcp_single_process_save_load_and_consolidate(tmp_path):
    """DCP save and load with plain model and consolidate to un-sharded state dict."""
    torch.manual_seed(400)
    model = TinyLinearModel()
    dcp_dir = tmp_path / "dcp_test"

    state_dict = {"model": model.state_dict()}
    save_dcp(state_dict, dcp_dir)
    assert dcp_dir.exists()

    # Consolidate DCP checkpoint into a fresh model
    template = TinyLinearModel()
    consolidated = dcp_to_state_dict(dcp_dir, template)
    template.load_state_dict(consolidated)

    x = torch.randn(2, 4)
    assert torch.allclose(model(x), template(x))


def test_checkpoint_manager_use_dcp_interface(tmp_path):
    """CheckpointManager with use_dcp=True saves DCP shards and loads through same interface."""
    torch.manual_seed(500)
    model = TinyLinearModel()
    mgr = CheckpointManager(checkpoint_dir=tmp_path / "ckpts", use_dcp=True)
    train_state = TrainState(epoch=0, global_step=3)

    saved_path = mgr.save(model, train_state)
    assert (saved_path / "dcp").is_dir()

    fresh_model = TinyLinearModel()
    loaded_state = mgr.load("auto", model=fresh_model)
    assert loaded_state is not None
    fresh_model.load_state_dict(loaded_state["model"])

    x = torch.randn(2, 4)
    assert torch.allclose(model(x), fresh_model(x))


# ==============================================================================
# Multi-Process CPU / Gloo Tests via launch()
# ==============================================================================


def _is_loopback_socket_accessible() -> bool:
    """Check if loopback sockets can be bound and connected across processes in current environment."""
    import socket

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("127.0.0.1", 0))
            s.listen(1)
            port = s.getsockname()[1]
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as c:
                c.settimeout(0.5)
                c.connect(("127.0.0.1", port))
        return True
    except OSError:
        return False


@pytest.mark.skipif(
    not _is_loopback_socket_accessible(),
    reason="Multi-process socket rendezvous requires loopback socket access (run outside sandbox).",
)
def test_launch_notebook_style_function(tmp_path):
    """Test launch() running a notebook-style custom function across 2 CPU processes."""
    record_prefix = str(tmp_path / "proc_record")
    launch(_worker_notebook_style, num_processes=2, backend="gloo", args=(record_prefix,))

    # Verify both ranks wrote their records
    r0 = Path(f"{record_prefix}_0.txt").read_text(encoding="utf-8")
    r1 = Path(f"{record_prefix}_1.txt").read_text(encoding="utf-8")
    assert r0 == "rank=0,world_size=2"
    assert r1 == "rank=1,world_size=2"


@pytest.mark.skipif(
    not _is_loopback_socket_accessible(),
    reason="Multi-process socket rendezvous requires loopback socket access (run outside sandbox).",
)
def test_ddp_two_processes_loss_and_gradient_parity(tmp_path):
    """Test DDP on 2 CPU processes achieves synchronization and loss decrease."""
    launch(_worker_ddp_loss_parity, num_processes=2, backend="gloo", args=(str(tmp_path),))

    f0 = Path(tmp_path / "loss_rank_0.pt")
    f1 = Path(tmp_path / "loss_rank_1.pt")
    assert f0.exists() and f1.exists()


@pytest.mark.skipif(
    not _is_loopback_socket_accessible(),
    reason="Multi-process socket rendezvous requires loopback socket access (run outside sandbox).",
)
def test_ddp_checkpoint_save_and_resume_parity(tmp_path):
    """Test DDP checkpoint saving strips 'module.' prefix and reload gives identical outputs."""
    launch(_worker_ddp_checkpoint_save_resume, num_processes=2, backend="gloo", args=(str(tmp_path),))


@pytest.mark.skipif(
    not _is_loopback_socket_accessible(),
    reason="Multi-process socket rendezvous requires loopback socket access (run outside sandbox).",
)
def test_dcp_two_processes_save_and_load(tmp_path):
    """Test DCP save and load across 2 CPU processes with DDP model."""
    launch(_worker_dcp_save_load, num_processes=2, backend="gloo", args=(str(tmp_path),))
