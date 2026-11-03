"""Tests for structured pretty output display manager, Rich integration, and verbosity modes."""

import io
import re
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from stackformer.training.config import TrainConfig
from stackformer.training.display import DisplayManager, is_in_notebook, strip_ansi
from stackformer.training.trainer import Trainer

ANSI_REGEX = re.compile(r"\x1b\[[0-9;]*[a-zA-Z]")


def test_strip_ansi():
    """Verify ANSI escape sequences are completely stripped."""
    colored_text = "\x1b[31mError:\x1b[0m \x1b[1;32mSuccess\x1b[0m"
    assert strip_ansi(colored_text) == "Error: Success"


def test_display_plain_mode_no_ansi():
    """Verify display='plain' outputs clean text without any ANSI escape codes."""
    buf = io.StringIO()
    display = DisplayManager(mode="plain", verbosity="normal", file=buf)
    assert display.mode == "plain"

    display.print_header("Test Title", subtitle="Test Subtitle")
    display.print_config({"lr": 1e-3, "epochs": 5})
    display.print_epoch_summary(epoch=1, max_epochs=5, train_loss=0.4567, metrics={"acc": 0.89})
    display.print_eval_table({"val_loss": 0.4321, "val_acc": 0.91})

    output = buf.getvalue()
    assert "Test Title" in output
    assert "Test Subtitle" in output
    assert "lr" in output
    assert "0.4567" in output
    assert "val_loss" in output
    assert not ANSI_REGEX.search(output), f"Found unexpected ANSI escapes in plain mode: {output!r}"


def test_display_rich_mode_fallback_or_render():
    """Verify display='rich' either renders via Rich or falls back to plain cleanly."""
    buf = io.StringIO()
    display = DisplayManager(mode="rich", verbosity="normal", file=buf)
    display.print_header("Rich Header")
    display.print_config(TrainConfig(lr=2e-4, max_epochs=3))
    output = buf.getvalue()
    assert "Rich Header" in output
    assert "2e-04" in output or "0.0002" in output


def test_display_verbosity_quiet():
    """Verify verbosity='quiet' produces zero output across all methods."""
    buf = io.StringIO()
    display = DisplayManager(mode="plain", verbosity="quiet", file=buf)
    assert display.is_quiet

    display.log("Should not appear")
    display.print_header("Hidden Header")
    display.print_config({"key": "val"})
    display.print_epoch_summary(epoch=1, max_epochs=2, train_loss=1.0)
    display.print_eval_table({"acc": 0.9})
    display.print_step(step=1, loss=0.5, lr=1e-3)

    pbar = display.progress_bar(range(10), desc="Hidden Bar")
    assert not pbar.enabled

    assert buf.getvalue() == "", f"Expected empty output for quiet verbosity, got: {buf.getvalue()!r}"


def test_display_verbosity_verbose():
    """Verify verbosity='verbose' outputs per-step step logs."""
    buf = io.StringIO()
    display = DisplayManager(mode="plain", verbosity="verbose", file=buf)
    assert display.is_verbose

    display.print_step(step=42, loss=0.1234, lr=3e-4, grad_norm=0.98)
    output = buf.getvalue()
    assert "Step 42" in output
    assert "0.1234" in output
    assert "3.00e-04" in output or "0.0003" in output
    assert "grad_norm=0.98" in output


def test_display_rank_zero_only(monkeypatch):
    """Verify non-zero ranks are completely silent."""
    import stackformer.training.display as display_mod

    # Simulate non-zero rank
    monkeypatch.setattr(display_mod, "is_rank_zero", lambda: False)

    buf = io.StringIO()
    display = DisplayManager(mode="plain", verbosity="verbose", rank_zero_only=True, file=buf)
    assert not display.is_active

    display.log("Non-zero rank log")
    display.print_header("Rank 1 Header")
    display.print_config({"key": "val"})
    display.print_step(step=1, loss=0.5, lr=1e-3)
    pbar = display.progress_bar(range(5))
    assert not pbar.enabled

    assert buf.getvalue() == "", "Non-zero rank must produce no console output"


def test_display_non_tty_suppresses_ansi():
    """Verify non-TTY file streams suppress ANSI escape codes even when rich is active."""
    buf = io.StringIO()
    display = DisplayManager(mode="auto", verbosity="normal", file=buf)
    display.print_header("Banner")
    display.print_eval_table({"metric_a": 10.5})
    output = buf.getvalue()
    assert not ANSI_REGEX.search(output), f"Found ANSI escapes in non-TTY output: {output!r}"


def test_display_notebook_detection():
    """Verify notebook detection returns a boolean without throwing exceptions."""
    result = is_in_notebook()
    assert isinstance(result, bool)


def test_config_display_and_verbosity_validation():
    """Verify TrainConfig validates verbosity and display settings."""
    cfg = TrainConfig(verbosity="quiet", display="plain")
    assert cfg.verbosity == "quiet"
    assert cfg.display == "plain"

    with pytest.raises(ValueError, match="Unsupported verbosity"):
        TrainConfig(verbosity="invalid_verbosity")

    with pytest.raises(ValueError, match="Unsupported display"):
        TrainConfig(display="invalid_display")


def test_trainer_integration_with_display(tmp_path):
    """Verify Trainer integrates cleanly with DisplayManager and respects quiet verbosity."""
    torch.manual_seed(42)
    model = nn.Linear(4, 2)
    ds = TensorDataset(torch.randn(16, 4), torch.randint(0, 2, (16,)))
    loader = DataLoader(ds, batch_size=4)

    # Quiet run produces no output
    cfg = TrainConfig(
        max_epochs=1,
        device="cpu",
        verbosity="quiet",
        display="plain",
        checkpoint_dir=str(tmp_path),
    )
    trainer = Trainer(model, config=cfg, train_loader=loader)
    assert trainer.display.is_quiet
    state = trainer.train()
    assert state.global_step == 4
