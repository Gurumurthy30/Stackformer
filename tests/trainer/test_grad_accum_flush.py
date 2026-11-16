import copy

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from stackformer import GPT2, ModelConfig
from stackformer.training import TrainConfig, Trainer


def test_grad_accum_epoch_end_flush_steps_and_clears_grad(tmp_path):
    """Finding #35: Partial accumulation window at epoch end must step and clear gradients."""
    cfg_model = ModelConfig(
        vocab_size=32,
        num_layers=1,
        embed_dim=16,
        num_heads=2,
        seq_len=8,
        hidden_dim=32,
        dropout=0.0,
    )
    model = GPT2(cfg_model)

    # 3 micro-batches in the dataset, batch_size=1
    x = torch.randint(0, 32, (3, 8))
    loader = DataLoader(TensorDataset(x, x), batch_size=1, shuffle=False)

    # grad_accum_steps=2 -> micro-batches 1 & 2 form step 1; micro-batch 3 is a partial window of 1
    # Without fix: 1 step in epoch 1, leftover gradient leaks into epoch 2
    # With fix: 2 steps in epoch 1 (step 1 after 2 batches, step 2 after flushing 1 batch)
    cfg = TrainConfig(
        max_epochs=1,
        grad_accum_steps=2,
        optimizer="sgd",
        scheduler="none",
        lr=0.01,
        device="cpu",
        checkpoint_dir=str(tmp_path),
    )
    trainer = Trainer(model=model, config=cfg, train_loader=loader)
    state = trainer.train()

    # Must have completed 2 optimizer steps
    assert state.global_step == 2, f"Expected 2 steps after epoch flush, got {state.global_step}"
    # All gradients must be cleared (None or zero)
    for p in model.parameters():
        assert p.grad is None or (p.grad == 0).all(), "Gradients must not leak after epoch end"


def test_grad_accum_flush_rescaling_parity(tmp_path):
    """Finding #35: Partial window of k micro-batches must be rescaled by accum_steps/k."""
    torch.manual_seed(42)
    cfg_model = ModelConfig(
        vocab_size=32,
        num_layers=1,
        embed_dim=16,
        num_heads=2,
        seq_len=8,
        hidden_dim=32,
        dropout=0.0,
    )
    m_accum = GPT2(cfg_model)
    m_direct = GPT2(cfg_model)
    init_state = copy.deepcopy(m_accum.state_dict())
    m_direct.load_state_dict(init_state)

    # 1 micro-batch dataset
    x = torch.randint(0, 32, (1, 8))
    loader = DataLoader(TensorDataset(x, x), batch_size=1, shuffle=False)

    # Run 1: 1 batch with grad_accum_steps=2 (flushed at epoch end, rescaled by 2/1)
    cfg_accum = TrainConfig(
        max_epochs=1,
        grad_accum_steps=2,
        optimizer="sgd",
        scheduler="none",
        lr=0.01,
        device="cpu",
        checkpoint_dir=str(tmp_path / "accum"),
    )
    t_accum = Trainer(model=m_accum, config=cfg_accum, train_loader=loader)
    t_accum.train()

    # Run 2: 1 batch with grad_accum_steps=1 (standard single step)
    cfg_direct = TrainConfig(
        max_epochs=1,
        grad_accum_steps=1,
        optimizer="sgd",
        scheduler="none",
        lr=0.01,
        device="cpu",
        checkpoint_dir=str(tmp_path / "direct"),
    )
    t_direct = Trainer(model=m_direct, config=cfg_direct, train_loader=loader)
    t_direct.train()

    # Flushed step with grad_accum_steps=2 rescaled by 2/1 must match grad_accum_steps=1
    for p_accum, p_direct in zip(m_accum.parameters(), m_direct.parameters(), strict=True):
        assert torch.allclose(p_accum, p_direct, atol=1e-5), (
            "Partial accumulation flush did not properly rescale gradients"
        )
