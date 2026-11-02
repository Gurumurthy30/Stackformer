"""Tests for Step 2: Core Trainer on CPU (no distribution)."""

import copy
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from stackformer import GPT2, ModelConfig
from stackformer.training import Trainer, TrainConfig
from stackformer.training.losses import get_loss, register_loss


def create_tiny_gpt():
    torch.manual_seed(42)
    cfg = ModelConfig(
        vocab_size=64,
        embed_dim=32,
        num_layers=2,
        num_heads=2,
        seq_len=16,
        hidden_dim=64,
    )
    model = GPT2(cfg)
    model.create_weights(method="default")
    return model


# =====================================================================
# 1. Loss Registry Tests
# =====================================================================

def test_loss_registry_by_name():
    ce = get_loss("cross_entropy")
    assert callable(ce)
    mse = get_loss("mse")
    assert callable(mse)
    l1 = get_loss("l1")
    assert callable(l1)


def test_loss_registry_by_callable():
    custom_fn = lambda out, batch: (out - batch[1]).abs().mean()
    resolved = get_loss(custom_fn)
    assert resolved is custom_fn


def test_loss_registry_unknown_name_error():
    with pytest.raises(ValueError) as excinfo:
        get_loss("non_existent_loss_function")
    assert "Unknown loss" in str(excinfo.value)
    assert "cross_entropy" in str(excinfo.value)


def test_loss_registry_custom_registration():
    @register_loss("dummy_loss_test")
    def dummy_loss(out, batch):
        return torch.tensor(0.5)

    fn = get_loss("dummy_loss_test")
    assert fn(None, None).item() == 0.5


# =====================================================================
# 2. TrainConfig Validation & Loose Kwargs Tests
# =====================================================================

def test_train_config_validation():
    # Valid
    cfg = TrainConfig(max_epochs=2, lr=1e-3)
    assert cfg.max_epochs == 2

    # Invalid lr
    with pytest.raises(ValueError, match="'lr' must be positive"):
        TrainConfig(lr=-0.1)

    # Invalid optimizer
    with pytest.raises(ValueError, match="Unsupported optimizer"):
        TrainConfig(optimizer="unknown_opt")

    # Invalid scheduler
    with pytest.raises(ValueError, match="Unsupported scheduler"):
        TrainConfig(scheduler="unknown_sched")

    # Invalid epochs and steps
    with pytest.raises(ValueError, match="At least one of 'max_epochs' or 'max_steps'"):
        TrainConfig(max_epochs=None, max_steps=None)


def test_train_config_serialization():
    cfg = TrainConfig(max_epochs=3, max_steps=100, lr=1e-4, optimizer="adam")
    d = cfg.to_dict()
    assert d["max_epochs"] == 3
    assert d["lr"] == 1e-4
    loaded = TrainConfig.from_dict(d)
    assert loaded.max_epochs == 3
    assert loaded.lr == 1e-4
    assert loaded.optimizer == "adam"


def test_trainer_loose_kwargs_override():
    model = create_tiny_gpt()
    cfg = TrainConfig(lr=1e-4, max_epochs=2)
    # Loose kwarg lr=5e-3 overrides config lr=1e-4
    trainer = Trainer(model, cfg, lr=5e-3)
    assert trainer.config.lr == 5e-3
    assert trainer.config.max_epochs == 2


def test_device_auto_resolution():
    cfg = TrainConfig(device="auto")
    resolved = cfg.resolve_device()
    assert isinstance(resolved, torch.device)
    if not torch.cuda.is_available():
        assert resolved.type == "cpu"


# =====================================================================
# 3. Loss Parity: Trainer vs Plain PyTorch Loop
# =====================================================================

def test_loss_parity_with_plain_pytorch_loop():
    """Verify Trainer matches manual PyTorch loop step-for-step."""
    torch.manual_seed(1234)

    # Synthetic data
    x = torch.randint(0, 64, (8, 16))
    dataset = TensorDataset(x, x)
    loader = DataLoader(dataset, batch_size=4, shuffle=False)

    # Model A: Trained manually
    model_a = create_tiny_gpt()
    # Model B: Trained via Trainer
    model_b = create_tiny_gpt()
    model_b.load_state_dict(model_a.state_dict())

    # Plain loop: exactly 1 epoch (2 batches of 4)
    opt_a = torch.optim.AdamW(model_a.parameters(), lr=1e-3, weight_decay=0.0, betas=(0.9, 0.95), eps=1e-8)
    loss_fn_ce = torch.nn.CrossEntropyLoss()

    model_a.train()
    plain_losses = []
    for bx, by in loader:
        out = model_a(bx)
        shift_logits = out[:, :-1, :].contiguous().view(-1, 64)
        shift_labels = by[:, 1:].contiguous().view(-1)
        loss = loss_fn_ce(shift_logits, shift_labels)
        plain_losses.append(loss.item())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model_a.parameters(), 1.0)
        opt_a.step()
        opt_a.zero_grad(set_to_none=True)

    # Trainer loop
    cfg = TrainConfig(
        max_epochs=1,
        max_steps=None,
        lr=1e-3,
        weight_decay=0.0,
        optimizer="adamw",
        scheduler="none",
        grad_accum_steps=1,
        max_grad_norm=1.0,
        mixed_precision="no",
        device="cpu",
    )
    # Use custom optimizer with identical all-parameters group to match opt_a
    opt_b = torch.optim.AdamW(model_b.parameters(), lr=1e-3, weight_decay=0.0, betas=(0.9, 0.95), eps=1e-8)
    trainer = Trainer(model_b, cfg, train_loader=loader, optimizer=opt_b)
    trainer.train()

    # Compare parameters
    for p_a, p_b in zip(model_a.parameters(), model_b.parameters()):
        assert torch.allclose(p_a, p_b, atol=1e-5), "Parameter mismatch after training"


# =====================================================================
# 4. Max Steps vs Max Epochs Stopping
# =====================================================================

def test_stopping_max_steps_before_max_epochs():
    model = create_tiny_gpt()
    x = torch.randint(0, 64, (20, 16))
    loader = DataLoader(TensorDataset(x, x), batch_size=2)  # 10 batches/epoch

    # max_epochs=2 (20 steps total), but max_steps=5 -> should stop at 5
    cfg = TrainConfig(max_epochs=2, max_steps=5, lr=1e-3, scheduler="none")
    trainer = Trainer(model, cfg, train_loader=loader)
    state = trainer.train()

    assert state.global_step == 5
    assert state.epoch == 0


def test_stopping_max_epochs_before_max_steps():
    model = create_tiny_gpt()
    x = torch.randint(0, 64, (20, 16))
    loader = DataLoader(TensorDataset(x, x), batch_size=2)  # 10 batches/epoch

    # max_epochs=1 (10 steps total), max_steps=50 -> should stop at epoch 1 (step 10)
    cfg = TrainConfig(max_epochs=1, max_steps=50, lr=1e-3, scheduler="none")
    trainer = Trainer(model, cfg, train_loader=loader)
    state = trainer.train()

    assert state.global_step == 10
    assert state.epoch == 0


def test_stopping_max_steps_only():
    model = create_tiny_gpt()
    x = torch.randint(0, 64, (10, 16))
    loader = DataLoader(TensorDataset(x, x), batch_size=2)  # 5 batches/epoch

    # max_epochs=None, max_steps=12 -> loops through ~2.4 epochs to hit 12 steps
    cfg = TrainConfig(max_epochs=None, max_steps=12, lr=1e-3, scheduler="none")
    trainer = Trainer(model, cfg, train_loader=loader)
    state = trainer.train()

    assert state.global_step == 12
    assert state.epoch >= 2


# =====================================================================
# 5. Eval Frequency and Max Eval Steps
# =====================================================================

def test_eval_frequency_and_max_eval_steps():
    model = create_tiny_gpt()
    x_train = torch.randint(0, 64, (20, 16))
    x_val = torch.randint(0, 64, (30, 16))

    train_dl = DataLoader(TensorDataset(x_train, x_train), batch_size=2)
    val_dl = DataLoader(TensorDataset(x_val, x_val), batch_size=2)  # 15 batches

    eval_calls = []

    class MockTrainer(Trainer):
        def evaluate(self):
            eval_calls.append(self.state.global_step)
            return super().evaluate()

    # Eval every 3 steps, capped at 4 eval batches
    cfg = TrainConfig(
        max_epochs=1,
        max_steps=9,
        eval_every_n_steps=3,
        eval_every_n_epochs=None,
        max_eval_steps=4,
        scheduler="none",
    )
    trainer = MockTrainer(model, cfg, train_loader=train_dl, val_loader=val_dl)
    trainer.train()

    # Evaluated at steps 3, 6, 9
    assert eval_calls == [3, 6, 9]


# =====================================================================
# 6. Gradient Accumulation Equivalence
# =====================================================================

def test_gradient_accumulation_parity():
    """Verify 2 micro-batches with grad_accum_steps=2 matches single large batch."""
    torch.manual_seed(999)

    x1 = torch.randint(0, 64, (2, 16))
    x2 = torch.randint(0, 64, (2, 16))
    x_full = torch.cat([x1, x2], dim=0)  # batch size 4

    model_single = create_tiny_gpt()
    init_state = copy.deepcopy(model_single.state_dict())

    # Run 1: Single step with batch size 4
    opt_single = torch.optim.SGD(model_single.parameters(), lr=0.01)
    out_full = model_single(x_full)
    loss_full = torch.nn.functional.cross_entropy(
        out_full[:, :-1, :].reshape(-1, 64),
        x_full[:, 1:].reshape(-1),
    )
    loss_full.backward()
    opt_single.step()

    # Run 2: Two micro-batches of 2 with grad_accum_steps=2 via Trainer
    model_accum = create_tiny_gpt()
    model_accum.load_state_dict(init_state)

    loader = DataLoader(TensorDataset(x_full, x_full), batch_size=2, shuffle=False)
    opt_accum = torch.optim.SGD(model_accum.parameters(), lr=0.01)

    cfg = TrainConfig(
        max_epochs=1,
        grad_accum_steps=2,
        optimizer="sgd",
        scheduler="none",
        max_grad_norm=None,
        device="cpu",
    )
    trainer = Trainer(model_accum, cfg, train_loader=loader, optimizer=opt_accum)
    trainer.train()

    for p1, p2 in zip(model_single.parameters(), model_accum.parameters()):
        assert torch.allclose(p1, p2, atol=1e-5), "Gradient accumulation mismatch"


# =====================================================================
# 7. Loss Decrease Tests (LLM next-token shift & Classification)
# =====================================================================

def test_llm_shift_loss_decrease():
    """Verify tiny GPT2 trains and decreases loss on repeated sequences."""
    torch.manual_seed(42)
    model = create_tiny_gpt()

    # Repeated pattern
    pattern = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]] * 16)
    loader = DataLoader(TensorDataset(pattern, pattern), batch_size=4, shuffle=True)

    # Initial eval loss
    cfg = TrainConfig(max_epochs=5, lr=2e-3, optimizer="adamw", scheduler="none")
    trainer = Trainer(model, cfg, train_loader=loader, val_loader=loader)

    init_loss = trainer.evaluate()["val_loss"]
    trainer.train()
    final_loss = trainer.evaluate()["val_loss"]

    assert final_loss < init_loss, f"Expected loss to decrease: init={init_loss}, final={final_loss}"


def test_classification_loss_decrease():
    """Verify Trainer trains a standard classification model."""
    torch.manual_seed(42)
    # Simple MLP classifier
    model = nn.Sequential(
        nn.Linear(10, 32),
        nn.ReLU(),
        nn.Linear(32, 2),
    )
    # Synthetic separable data
    x = torch.randn(64, 10)
    y = (x[:, 0] > 0).long()
    loader = DataLoader(TensorDataset(x, y), batch_size=16)

    cfg = TrainConfig(max_epochs=5, lr=1e-2, loss="cross_entropy", scheduler="none")
    trainer = Trainer(model, cfg, train_loader=loader, val_loader=loader)

    init_loss = trainer.evaluate()["val_loss"]
    trainer.train()
    final_loss = trainer.evaluate()["val_loss"]

    assert final_loss < init_loss, f"Expected classification loss to decrease: {init_loss} -> {final_loss}"


# =====================================================================
# 8. Loss Mode Resolution & Ambiguous Case Tests
# =====================================================================

def test_cross_entropy_loss_mode_paths():
    """Test 'auto', 'lm', and 'classification' modes and ambiguous 3D+2D input case."""
    from stackformer.training.losses import cross_entropy_loss

    # 1. Standard classification: 2D logits (B, C), 1D targets (B,)
    logits_2d = torch.tensor([[2.0, 1.0], [0.5, 3.0]])
    labels_1d = torch.tensor([0, 1])

    loss_auto_cls = cross_entropy_loss(logits_2d, (None, labels_1d), loss_mode="auto")
    loss_explicit_cls = cross_entropy_loss(logits_2d, (None, labels_1d), loss_mode="classification")
    expected_cls = torch.nn.functional.cross_entropy(logits_2d, labels_1d)
    assert torch.allclose(loss_auto_cls, expected_cls)
    assert torch.allclose(loss_explicit_cls, expected_cls)

    # 2. Ambiguous shape: 3D logits (B, T, V), 2D targets (B, T)
    # Could be either causal LM (needs shift) OR dense token classification / masked LM (no shift).
    logits_3d = torch.randn(2, 4, 8)
    labels_2d = torch.randint(0, 8, (2, 4))

    # Shifted LM loss: logits[:, :-1, :] vs labels[:, 1:]
    expected_lm = torch.nn.functional.cross_entropy(
        logits_3d[:, :-1, :].reshape(-1, 8),
        labels_2d[:, 1:].reshape(-1),
    )
    # Unshifted classification loss: logits[:, :, :] vs labels[:, :]
    expected_no_shift = torch.nn.functional.cross_entropy(
        logits_3d.reshape(-1, 8),
        labels_2d.reshape(-1),
    )

    loss_auto = cross_entropy_loss(logits_3d, (None, labels_2d), loss_mode="auto")
    loss_lm = cross_entropy_loss(logits_3d, (None, labels_2d), loss_mode="lm")
    loss_dense_cls = cross_entropy_loss(logits_3d, (None, labels_2d), loss_mode="classification")

    # 'auto' defaults to causal LM shift on 3D+2D
    assert torch.allclose(loss_auto, expected_lm)
    assert torch.allclose(loss_lm, expected_lm)

    # Explicit 'classification' overrides shift and applies dense token CE without shift
    assert torch.allclose(loss_dense_cls, expected_no_shift)
    assert not torch.allclose(loss_lm, loss_dense_cls), "Shifted and unshifted losses should differ"

    # 3. Invalid mode raises ValueError
    with pytest.raises(ValueError, match="'loss_mode' must be one of"):
        TrainConfig(loss_mode="invalid")

