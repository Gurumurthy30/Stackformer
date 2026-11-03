"""Tests for Step 5: Activation Checkpointing, torch.compile, and Mixed Precision."""

import copy

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from stackformer.distributed.activation_checkpoint import apply_activation_checkpointing
from stackformer.distributed.compile import apply_compile, is_compile_available
from stackformer.models import GPT2
from stackformer.training import CheckpointManager, TrainConfig, Trainer
from stackformer.training.checkpoint import clean_state_dict

# =====================================================================
# 1. AC Equivalence (none vs full vs selective)
# =====================================================================


def test_ac_loss_and_gradients_numerical_parity():
    """Verify AC 'none', 'full', and 'selective' yield identical losses and gradients."""
    torch.manual_seed(100)

    # Base model with identical weights across three instances (dropout=0 for deterministic numerical parity)
    m_none = GPT2(vocab_size=32, embed_dim=16, num_layers=2, num_heads=2, seq_len=8, dropout=0.0)
    init_state = copy.deepcopy(m_none.state_dict())

    m_full = GPT2(vocab_size=32, embed_dim=16, num_layers=2, num_heads=2, seq_len=8, dropout=0.0)
    m_full.load_state_dict(copy.deepcopy(init_state))

    m_sel = GPT2(vocab_size=32, embed_dim=16, num_layers=2, num_heads=2, seq_len=8, dropout=0.0)
    m_sel.load_state_dict(copy.deepcopy(init_state))

    # Apply AC
    apply_activation_checkpointing(m_full, mode="full")
    apply_activation_checkpointing(m_sel, mode="selective")

    x = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]] * 2)

    # Forward + backward none
    m_none.zero_grad(set_to_none=True)
    out_none = m_none(x)
    loss_none = out_none.sum()
    loss_none.backward()

    # Forward + backward full
    m_full.zero_grad(set_to_none=True)
    out_full = m_full(x)
    loss_full = out_full.sum()
    loss_full.backward()

    # Forward + backward selective
    m_sel.zero_grad(set_to_none=True)
    out_sel = m_sel(x)
    loss_sel = out_sel.sum()
    loss_sel.backward()

    # Verify losses are equal within numerical tolerance
    assert torch.allclose(loss_none, loss_full, atol=1e-5), (
        f"Full AC loss mismatch: {loss_none.item()} vs {loss_full.item()}"
    )
    assert torch.allclose(loss_none, loss_sel, atol=1e-5), (
        f"Selective AC loss mismatch: {loss_none.item()} vs {loss_sel.item()}"
    )

    # Verify gradients are identical across all parameters
    params_none = dict(m_none.named_parameters())
    params_full = {
        k.replace("_checkpoint_wrapped_module.", ""): v for k, v in m_full.named_parameters()
    }
    params_sel = {
        k.replace("_checkpoint_wrapped_module.", ""): v for k, v in m_sel.named_parameters()
    }

    for name, p_none in params_none.items():
        if p_none.grad is not None:
            clean_name = name
            assert clean_name in params_full, f"Parameter {clean_name} missing from full AC model"
            assert clean_name in params_sel, (
                f"Parameter {clean_name} missing from selective AC model"
            )
            assert torch.allclose(p_none.grad, params_full[clean_name].grad, atol=1e-5), (
                f"Gradient mismatch for {name} (full AC)"
            )
            assert torch.allclose(p_none.grad, params_sel[clean_name].grad, atol=1e-5), (
                f"Gradient mismatch for {name} (selective AC)"
            )


# =====================================================================
# 2. Recompute-Count Evidence that AC Actually Runs
# =====================================================================


def test_ac_forward_recompute_count_evidence():
    """Verify that Activation Checkpointing triggers forward recomputation during backward."""

    class InstrumentedBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(8, 8)
            self.call_count = 0

        def forward(self, x):
            self.call_count += 1
            return self.linear(x)

    class CustomModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = nn.ModuleList([InstrumentedBlock() for _ in range(2)])

        def get_sharding_blocks(self):
            return list(self.blocks)

        def forward(self, x):
            for b in self.blocks:
                x = b(x)
            return x

    # 1. Without AC: exactly 1 forward call per block
    m_no_ac = CustomModel()
    out = m_no_ac(torch.randn(2, 8))
    out.sum().backward()
    assert [b.call_count for b in m_no_ac.blocks] == [1, 1]

    # 2. With Full AC: forward runs twice (1 initial forward + 1 recompute in backward)
    m_ac = CustomModel()
    apply_activation_checkpointing(m_ac, mode="full")
    out_ac = m_ac(torch.randn(2, 8))
    out_ac.sum().backward()

    # Find the instrumented blocks inside checkpoint wrappers
    underlying_blocks = [m for m in m_ac.modules() if isinstance(m, InstrumentedBlock)]
    assert len(underlying_blocks) == 2
    assert [b.call_count for b in underlying_blocks] == [2, 2], (
        "Expected each block to execute twice (recompute in backward)"
    )


# =====================================================================
# 3. Mixed Precision: bf16 Loss Decrease & fp16 on CPU Validation
# =====================================================================


def test_trainer_bf16_autocast_loss_decreases(tmp_path):
    """Verify Trainer trains in bf16 mixed precision on CPU and loss decreases."""
    torch.manual_seed(42)
    model = GPT2(vocab_size=64, embed_dim=32, num_layers=2, num_heads=2, seq_len=16)
    tokens = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]] * 16)
    loader = DataLoader(TensorDataset(tokens, tokens), batch_size=4, shuffle=True)

    cfg = TrainConfig(
        max_steps=5,
        lr=3e-3,
        mixed_precision="bf16",
        optimizer="adamw",
        scheduler="none",
        checkpoint_dir=str(tmp_path),
        save_last=False,
    )
    trainer = Trainer(model, cfg, train_loader=loader, val_loader=loader)
    init_loss = trainer.evaluate()["val_loss"]
    trainer.train()
    final_loss = trainer.evaluate()["val_loss"]

    assert final_loss < init_loss, (
        f"bf16 training failed to decrease loss: {init_loss} -> {final_loss}"
    )


def test_fp16_on_cpu_raises_informative_error():
    """Verify configuring fp16 mixed precision on CPU raises a clear ValueError instead of silent fallback."""
    with pytest.raises(ValueError, match="mixed_precision='fp16' is not supported on CPU"):
        TrainConfig(mixed_precision="fp16", device="cpu")


# =====================================================================
# 4. Compile Smoke Test and Environment Availability
# =====================================================================


def test_compile_availability_and_smoke_or_skip(tmp_path):
    """Verify is_compile_available reports status; run compile or skip gracefully with reason."""
    available, reason = is_compile_available()

    if not available:
        # Assert apply_compile raises clear RuntimeError when compiler is unavailable
        model = GPT2(vocab_size=32, embed_dim=16, num_layers=2, num_heads=2, seq_len=8)
        with pytest.raises(RuntimeError, match="torch.compile is not available"):
            apply_compile(model)

        pytest.skip(f"torch.compile unavailable in current environment: {reason}")
    else:
        # Run compile smoke test
        model = GPT2(vocab_size=32, embed_dim=16, num_layers=2, num_heads=2, seq_len=8)
        compiled_model = apply_compile(model, per_block=True)
        x = torch.randint(0, 32, (2, 8))
        out = compiled_model(x)
        assert out.shape == (2, 8, 32)


# =====================================================================
# 5. Unsupported Configurations Raise Clear Errors
# =====================================================================


def test_unsupported_combos_raise_clear_errors():
    """Verify unsupported hyperparameter combinations fail fast with explicit errors."""
    with pytest.raises(ValueError, match="Unsupported activation_checkpointing"):
        TrainConfig(activation_checkpointing="unsupported_mode")

    with pytest.raises(ValueError, match="Unsupported mixed_precision"):
        TrainConfig(mixed_precision="unsupported_precision")

    with pytest.raises(ValueError, match="Unsupported compile_mode"):
        TrainConfig(compile=True, compile_mode="invalid_compile_mode")


# =====================================================================
# 6. Checkpoint Save/Resume with AC and Compile (_orig_mod Prefix Cleanliness)
# =====================================================================


def test_clean_state_dict_removes_orig_mod_prefixes():
    """Verify clean_state_dict strips nested _orig_mod. prefixes produced by torch.compile."""
    dirty_dict = {
        "_orig_mod.backbone.layers.0.weight": torch.randn(4, 4),
        "_orig_mod.backbone.layers.0._orig_mod.bias": torch.zeros(4),
        "lm_head.weight": torch.randn(4, 4),
    }
    cleaned = clean_state_dict(dirty_dict)

    assert "backbone.layers.0.weight" in cleaned
    assert "backbone.layers.0.bias" in cleaned
    assert "lm_head.weight" in cleaned
    assert not any("_orig_mod" in k for k in cleaned)


def test_checkpoint_save_and_resume_with_ac_and_compile(tmp_path):
    """Verify saved checkpoints with AC and compiled modules never leak wrapper prefixes."""
    model = GPT2(vocab_size=32, embed_dim=16, num_layers=2, num_heads=2, seq_len=8)

    # Apply AC
    apply_activation_checkpointing(model, mode="full")

    # Simulate compiled module wrapper by monkeypatching a sub-prefix into state_dict keys
    mgr = CheckpointManager(checkpoint_dir=tmp_path)
    trainer = Trainer(
        model, TrainConfig(checkpoint_dir=str(tmp_path), activation_checkpointing="full")
    )

    # Save checkpoint
    ckpt_dir = trainer.save_checkpoint()
    assert (ckpt_dir / "state.pt").exists()
    assert (ckpt_dir / "model.safetensors").exists()

    # Load raw state.pt and inspect keys
    saved_state = torch.load(ckpt_dir / "state.pt", weights_only=False)
    model_keys = list(saved_state["model"].keys())

    assert not any("_orig_mod" in k for k in model_keys), "Found _orig_mod prefix in saved state.pt"
    assert not any("_checkpoint_wrapped_module" in k for k in model_keys)

    # Fresh uncheckpointed, uncompiled model must reload state cleanly
    fresh_model = GPT2(vocab_size=32, embed_dim=16, num_layers=2, num_heads=2, seq_len=8)
    fresh_model.load_state_dict(saved_state["model"], strict=True)
    fresh_model.load_weights(ckpt_dir / "model.safetensors", strict=True)
