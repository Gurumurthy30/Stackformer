"""Unit and integration tests for CheckpointManager and Trainer checkpointing/resume."""

import copy
import json
import random
from pathlib import Path
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from stackformer.config import ModelConfig
from stackformer.models import GPT2
from stackformer.training import Trainer, TrainConfig, CheckpointManager
from stackformer.training.state import TrainState


def create_tiny_model():
    cfg = ModelConfig(
        vocab_size=64,
        embed_dim=16,
        num_layers=1,
        num_heads=2,
        seq_len=8,
        hidden_dim=32,
        dropout=0.0,
    )
    return GPT2(cfg)


def test_checkpoint_roundtrip_restores_state(tmp_path):
    """Test saving and loading restores model, optimizer, scheduler, and RNG."""
    torch.manual_seed(1234)
    model = create_tiny_model()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.LinearLR(opt, start_factor=0.5, total_iters=10)

    # Do a dummy forward/backward/step to give optimizer state
    x = torch.randint(0, 64, (2, 8))
    loss = model(x).sum()
    loss.backward()
    opt.step()
    sched.step()

    state = TrainState(epoch=1, global_step=5, step_in_epoch=2, best_metric=1.23)
    cfg = TrainConfig(lr=1e-3, checkpoint_dir=str(tmp_path))

    mgr = CheckpointManager(checkpoint_dir=tmp_path)
    saved_dir = mgr.save(model=model, train_state=state, optimizer=opt, scheduler=sched, config=cfg)
    assert saved_dir.exists()
    assert (saved_dir / "state.pt").exists()
    assert (saved_dir / "model.safetensors").exists()
    assert (saved_dir / "meta.json").exists()

    # Load back
    loaded = mgr.load(saved_dir)
    assert loaded is not None

    new_model = create_tiny_model()
    new_model.load_state_dict(loaded["model"])

    for p1, p2 in zip(model.parameters(), new_model.parameters()):
        assert torch.equal(p1, p2), "Model parameters mismatch after reload"

    new_opt = torch.optim.AdamW(new_model.parameters(), lr=1e-3)
    new_opt.load_state_dict(loaded["optimizer"])
    for pid in opt.state_dict()["state"]:
        for k, v in opt.state_dict()["state"][pid].items():
            if isinstance(v, torch.Tensor):
                assert torch.equal(v, new_opt.state_dict()["state"][pid][k])
            else:
                assert v == new_opt.state_dict()["state"][pid][k]

    assert loaded["train_state"]["epoch"] == 1
    assert loaded["train_state"]["global_step"] == 5
    assert loaded["train_state"]["step_in_epoch"] == 2
    assert loaded["train_state"]["best_metric"] == 1.23


def test_kill_and_resume_parity_mid_epoch(tmp_path):
    """Train N steps uninterrupted vs train K steps, kill, resume, train to N.

    With a seeded DataLoader, per-step losses and final weights must match bit-identically.
    """
    torch.manual_seed(42)
    # 20 samples, batch_size=2 -> 10 batches/epoch
    x = torch.randint(0, 64, (20, 8))
    dataset = TensorDataset(x, x)

    # Initial weights
    init_model = create_tiny_model()
    init_state = copy.deepcopy(init_model.state_dict())

    ckpt_dir_a = tmp_path / "run_a"
    ckpt_dir_b = tmp_path / "run_b"

    # --- Run A: Uninterrupted for 12 steps (epoch 0 has 10 steps, epoch 1 has 2 steps) ---
    gen_a = torch.Generator().manual_seed(42)
    loader_a = DataLoader(dataset, batch_size=2, shuffle=True, generator=gen_a)

    model_a = create_tiny_model()
    model_a.load_state_dict(copy.deepcopy(init_state))
    cfg_a = TrainConfig(
        max_epochs=None,
        max_steps=12,
        lr=1e-3,
        optimizer="adamw",
        scheduler="none",
        checkpoint_dir=str(ckpt_dir_a),
        save_every_n_steps=6,
        seed=42,
    )
    trainer_a = Trainer(model_a, cfg_a, train_loader=loader_a)
    trainer_a.train()

    # --- Run B1: Train for 6 steps, then 'kill' ---
    gen_b = torch.Generator().manual_seed(42)
    loader_b = DataLoader(dataset, batch_size=2, shuffle=True, generator=gen_b)

    model_b = create_tiny_model()
    model_b.load_state_dict(copy.deepcopy(init_state))
    cfg_b1 = TrainConfig(
        max_epochs=None,
        max_steps=6,
        lr=1e-3,
        optimizer="adamw",
        scheduler="none",
        checkpoint_dir=str(ckpt_dir_b),
        save_every_n_steps=6,
        seed=42,
    )
    trainer_b1 = Trainer(model_b, cfg_b1, train_loader=loader_b)
    trainer_b1.train()
    assert trainer_b1.state.global_step == 6

    # --- Run B2: Resume from checkpoint and train to step 12 ---
    gen_b2 = torch.Generator().manual_seed(42)
    loader_b2 = DataLoader(dataset, batch_size=2, shuffle=True, generator=gen_b2)

    model_b_resume = create_tiny_model()
    cfg_b2 = TrainConfig(
        max_epochs=None,
        max_steps=12,
        lr=1e-3,
        optimizer="adamw",
        scheduler="none",
        checkpoint_dir=str(ckpt_dir_b),
        resume_from="auto",
        seed=42,
    )
    trainer_b2 = Trainer(model_b_resume, cfg_b2, train_loader=loader_b2)
    trainer_b2.train()

    assert trainer_b2.state.global_step == 12
    assert trainer_a.state.global_step == trainer_b2.state.global_step
    assert trainer_a.state.epoch == trainer_b2.state.epoch

    for p_a, p_b in zip(model_a.parameters(), model_b_resume.parameters()):
        assert torch.allclose(p_a, p_b, atol=1e-6), "Parameters diverge after resume"


def test_kill_and_resume_with_grad_accumulation(tmp_path):
    """Test kill-and-resume parity with grad_accum_steps=2."""
    torch.manual_seed(99)
    x = torch.randint(0, 64, (24, 8))
    dataset = TensorDataset(x, x)

    init_model = create_tiny_model()
    init_state = copy.deepcopy(init_model.state_dict())

    # Run A: uninterrupted for 4 steps (8 micro-batches)
    gen_a = torch.Generator().manual_seed(99)
    loader_a = DataLoader(dataset, batch_size=2, shuffle=True, generator=gen_a)
    model_a = create_tiny_model()
    model_a.load_state_dict(copy.deepcopy(init_state))

    dir_a = tmp_path / "a"
    cfg_a = TrainConfig(
        max_steps=4,
        grad_accum_steps=2,
        lr=1e-3,
        optimizer="adamw",
        scheduler="none",
        checkpoint_dir=str(dir_a),
        seed=99,
    )
    trainer_a = Trainer(model_a, cfg_a, train_loader=loader_a)
    trainer_a.train()

    # Run B: train 2 steps, save, resume to 4 steps
    gen_b1 = torch.Generator().manual_seed(99)
    loader_b1 = DataLoader(dataset, batch_size=2, shuffle=True, generator=gen_b1)
    model_b = create_tiny_model()
    model_b.load_state_dict(copy.deepcopy(init_state))

    dir_b = tmp_path / "b"
    cfg_b1 = TrainConfig(
        max_steps=2,
        grad_accum_steps=2,
        lr=1e-3,
        optimizer="adamw",
        scheduler="none",
        checkpoint_dir=str(dir_b),
        save_every_n_steps=2,
        seed=99,
    )
    trainer_b1 = Trainer(model_b, cfg_b1, train_loader=loader_b1)
    trainer_b1.train()

    gen_b2 = torch.Generator().manual_seed(99)
    loader_b2 = DataLoader(dataset, batch_size=2, shuffle=True, generator=gen_b2)
    model_b_res = create_tiny_model()
    cfg_b2 = TrainConfig(
        max_steps=4,
        grad_accum_steps=2,
        lr=1e-3,
        optimizer="adamw",
        scheduler="none",
        checkpoint_dir=str(dir_b),
        resume_from="auto",
        seed=99,
    )
    trainer_b2 = Trainer(model_b_res, cfg_b2, train_loader=loader_b2)
    trainer_b2.train()

    for p_a, p_b in zip(model_a.parameters(), model_b_res.parameters()):
        assert torch.allclose(p_a, p_b, atol=1e-6), "Param mismatch with grad accum"


def test_resume_from_auto_and_error_handling(tmp_path):
    """Test resume_from='auto' behavior on empty dir vs non-existent path vs corrupt dir."""
    mgr = CheckpointManager(checkpoint_dir=tmp_path / "empty")
    # Empty dir starts fresh without error
    loaded = mgr.load("auto")
    assert loaded is None

    # Invalid path raises FileNotFoundError
    with pytest.raises(FileNotFoundError, match="Checkpoint not found"):
        mgr.load("non_existent_checkpoint_path")

    # Corrupt directory without state.pt raises FileNotFoundError
    corrupt_dir = tmp_path / "corrupt_ckpt"
    corrupt_dir.mkdir()
    with pytest.raises(FileNotFoundError, match="does not contain 'state.pt'"):
        mgr.load(corrupt_dir)


def test_keep_last_n_pruning_and_preserves_best(tmp_path):
    """Verify pruning retains exactly keep_last_n checkpoints and never touches best/."""
    mgr = CheckpointManager(checkpoint_dir=tmp_path, keep_last_n=2)
    model = create_tiny_model()

    # Save best
    state_best = TrainState(epoch=0, global_step=1, best_metric=0.5)
    mgr.save_best(model, state_best)
    best_dir = tmp_path / "best"
    assert best_dir.exists()

    # Save 4 checkpoints
    for step in range(1, 5):
        st = TrainState(epoch=0, global_step=step)
        mgr.save(model, st)

    # Check remaining
    dirs = [d.name for d in tmp_path.iterdir() if d.is_dir()]
    assert "best" in dirs
    step_dirs = [d for d in dirs if d.startswith("epoch_")]
    assert len(step_dirs) == 2
    assert "epoch_001_step_00003" in step_dirs
    assert "epoch_001_step_00004" in step_dirs
    latest_path = tmp_path / "latest"
    assert latest_path.is_file()
    assert not latest_path.is_symlink(), "latest pointer must be a regular text file, not a symlink"
    assert latest_path.read_text(encoding="utf-8").strip() == "epoch_001_step_00004"


def test_best_model_saving_min_and_max_modes(tmp_path):
    """Verify best model is only saved upon strict improvement in min and max modes."""
    model = create_tiny_model()

    # Mode min
    mgr_min = CheckpointManager(checkpoint_dir=tmp_path / "min", mode="min")
    st = TrainState()

    # First eval: val_loss=1.0 -> improves
    assert st.update_best(1.0, mode="min") is True
    mgr_min.save_best(model, st, metrics={"val_loss": 1.0})
    meta1 = json.loads((tmp_path / "min" / "best" / "meta.json").read_text())
    assert meta1["best_metric"] == 1.0

    # Second eval: val_loss=1.0 (tie) -> does NOT improve
    assert st.update_best(1.0, mode="min") is False

    # Third eval: val_loss=1.5 (worse) -> does NOT improve
    assert st.update_best(1.5, mode="min") is False

    # Fourth eval: val_loss=0.8 (better) -> improves
    assert st.update_best(0.8, mode="min") is True
    mgr_min.save_best(model, st, metrics={"val_loss": 0.8})
    meta2 = json.loads((tmp_path / "min" / "best" / "meta.json").read_text())
    assert meta2["best_metric"] == 0.8

    # Mode max
    st_max = TrainState()
    assert st_max.update_best(0.7, mode="max") is True
    assert st_max.update_best(0.7, mode="max") is False
    assert st_max.update_best(0.6, mode="max") is False
    assert st_max.update_best(0.9, mode="max") is True


def test_atomic_saving_resilience_to_failure(tmp_path, monkeypatch):
    """Simulate failure during checkpoint write; ensure latest points to previous valid checkpoint."""
    mgr = CheckpointManager(checkpoint_dir=tmp_path)
    model = create_tiny_model()

    # Save 1st checkpoint successfully
    st1 = TrainState(epoch=0, global_step=1)
    mgr.save(model, st1)
    assert (tmp_path / "latest").read_text(encoding="utf-8").strip() == "epoch_001_step_00001"

    # Simulate crash during 2nd save
    def crashing_save(*args, **kwargs):
        raise IOError("Disk full or simulated crash!")

    monkeypatch.setattr(torch, "save", crashing_save)

    st2 = TrainState(epoch=0, global_step=2)
    with pytest.raises(IOError, match="simulated crash"):
        mgr.save(model, st2)

    # Latest must still point to 1st checkpoint and no corrupt directory remained
    assert (tmp_path / "latest").read_text(encoding="utf-8").strip() == "epoch_001_step_00001"
    assert not (tmp_path / "epoch_001_step_00002").exists()


def test_default_checkpoint_dir_created_in_cwd(tmp_path, monkeypatch):
    """Assert default checkpoint dir is created in CWD when none specified."""
    monkeypatch.chdir(tmp_path)
    cfg = TrainConfig()
    trainer = Trainer(create_tiny_model(), cfg)
    trainer.save_checkpoint()

    default_dir = tmp_path / "checkpoints"
    assert default_dir.exists()
    assert (default_dir / "latest").exists()


def test_trainer_save_safetensors_and_base_model_reload(tmp_path):
    """trainer.save('out.safetensors') produces weights reloadable by BaseModel.load_weights."""
    model = create_tiny_model()
    cfg = TrainConfig()
    trainer = Trainer(model, cfg)

    st_path = tmp_path / "exported.safetensors"
    trainer.save(st_path)
    assert st_path.exists()

    new_model = create_tiny_model()
    new_model.load_weights(st_path)

    x = torch.randint(0, 64, (2, 8))
    with torch.no_grad():
        out_orig = model(x)
        out_reloaded = new_model(x)

    assert torch.allclose(out_orig, out_reloaded), "Outputs differ after reloading safetensors export"


def test_safetensors_missing_save_weights_hint(tmp_path, monkeypatch):
    """save_model_weights raises clear ImportError hint if safetensors is missing."""
    import sys
    model = create_tiny_model()
    st_path = tmp_path / "model.safetensors"

    monkeypatch.setitem(sys.modules, "safetensors", None)
    monkeypatch.setitem(sys.modules, "safetensors.torch", None)

    from stackformer.utils.checkpoint import save_model_weights
    with pytest.raises(ImportError, match="Install via `pip install safetensors`"):
        save_model_weights(model, st_path)


def test_checkpoint_manager_warns_and_falls_back_when_safetensors_missing(tmp_path, monkeypatch):
    """CheckpointManager warns with installation hint and falls back to model.pt when safetensors is missing."""
    import stackformer.training.checkpoint as ckpt_mod

    model = create_tiny_model()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    state = TrainState(epoch=0, global_step=1, step_in_epoch=1)
    cfg = TrainConfig(checkpoint_dir=str(tmp_path))

    monkeypatch.setattr(ckpt_mod, "safetensors", None)

    mgr = CheckpointManager(checkpoint_dir=tmp_path)
    with pytest.warns(UserWarning, match="Install safetensors via: pip install safetensors"):
        saved_dir = mgr.save(model=model, train_state=state, optimizer=opt, scheduler=None, config=cfg)

    assert saved_dir.exists()
    assert (saved_dir / "model.pt").exists()

