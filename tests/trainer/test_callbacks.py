"""Integration and unit tests for callbacks, metrics, and loggers working together."""

import csv
import json
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from stackformer.config import ModelConfig
from stackformer.models import GPT2
from stackformer.training import TrainConfig, Trainer
from stackformer.training.callbacks import Callback, EarlyStopping, LRMonitor
from stackformer.training.loggers import CSVLogger
from stackformer.training.metrics import Accuracy, Perplexity


def create_tiny_gpt():
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


class RecordingCallback(Callback):
    """Records the order of hook invocations."""

    def __init__(self):
        self.events = []

    def on_train_start(self, trainer):
        self.events.append("train_start")

    def on_epoch_start(self, trainer, epoch):
        self.events.append(f"epoch_start_{epoch}")

    def on_step_end(self, trainer, step, metrics):
        self.events.append(f"step_end_{step}")

    def on_epoch_end(self, trainer, epoch, metrics):
        self.events.append(f"epoch_end_{epoch}")

    def on_eval_end(self, trainer, metrics):
        self.events.append("eval_end")

    def on_checkpoint_save(self, trainer, checkpoint_dir):
        self.events.append("checkpoint_save")

    def on_train_end(self, trainer):
        self.events.append("train_end")


def test_callback_hook_order(tmp_path):
    """Verify correct invocation ordering of all lifecycle callback hooks."""
    model = create_tiny_gpt()
    x = torch.randint(0, 64, (4, 8))
    loader = DataLoader(TensorDataset(x, x), batch_size=2)  # 2 batches per epoch

    rec = RecordingCallback()
    cfg = TrainConfig(
        max_epochs=1,
        checkpoint_dir=str(tmp_path),
        save_every_n_epochs=1,
        eval_every_n_epochs=1,
        scheduler="none",
    )
    trainer = Trainer(model, cfg, train_loader=loader, val_loader=loader, callbacks=[rec])
    trainer.train()

    assert rec.events[0] == "train_start"
    assert rec.events[1] == "epoch_start_0"
    assert "step_end_1" in rec.events
    assert "step_end_2" in rec.events
    assert "checkpoint_save" in rec.events
    assert "eval_end" in rec.events
    assert rec.events[-1] == "train_end"


def test_early_stopping_patience(tmp_path):
    """EarlyStopping halts training after exactly `patience` non-improving evaluations."""
    model = create_tiny_gpt()
    x = torch.randint(0, 64, (8, 8))
    loader = DataLoader(TensorDataset(x, x), batch_size=2)  # 4 batches/epoch

    # Custom trainer that returns fixed or worsening validation loss
    eval_losses = [2.0, 2.5, 2.6, 2.7, 2.8]
    call_idx = 0

    class StaticLossTrainer(Trainer):
        def evaluate(self):
            nonlocal call_idx
            loss = eval_losses[min(call_idx, len(eval_losses) - 1)]
            call_idx += 1
            eval_metrics = {"val_loss": loss}
            for cb in self.callbacks:
                cb.on_eval_end(self, eval_metrics)
            return eval_metrics

    # Patience = 2: eval 0 is 2.0 (baseline), eval 1 is 2.5 (1st fail), eval 2 is 2.6 (2nd fail -> stops)
    es = EarlyStopping(monitor="val_loss", mode="min", patience=2)
    cfg = TrainConfig(
        max_epochs=10,
        eval_every_n_steps=1,
        eval_every_n_epochs=None,
        checkpoint_dir=str(tmp_path),
        scheduler="none",
    )
    trainer = StaticLossTrainer(model, cfg, train_loader=loader, val_loader=loader, callbacks=[es])
    state = trainer.train()

    assert state.should_stop is True
    # Initial eval + 2 failures = 3 evaluations total
    assert call_idx == 3


def test_lr_monitor_records_scheduled_lr(tmp_path):
    """LRMonitor logs scheduled learning rate to loggers at each step."""
    model = create_tiny_gpt()
    x = torch.randint(0, 64, (6, 8))
    loader = DataLoader(TensorDataset(x, x), batch_size=2)

    csv_logger = CSVLogger(log_dir=tmp_path, filename="lr_log.csv")
    lr_mon = LRMonitor()

    cfg = TrainConfig(
        max_epochs=1,
        lr=1e-3,
        scheduler="linear",
        checkpoint_dir=str(tmp_path),
    )
    trainer = Trainer(
        model,
        cfg,
        train_loader=loader,
        loggers=[csv_logger],
        callbacks=[lr_mon],
    )
    trainer.train()

    csv_file = tmp_path / "lr_log.csv"
    assert csv_file.exists()
    with open(csv_file, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    assert len(rows) > 0
    assert "lr" in rows[0]
    assert float(rows[0]["lr"]) > 0.0


def test_integration_gpt2_metrics_and_best_model(tmp_path):
    """Train tiny GPT2 for 2 epochs with Perplexity metric and best-model monitoring."""
    torch.manual_seed(42)
    model = create_tiny_gpt()
    x = torch.randint(0, 64, (8, 8))
    loader = DataLoader(TensorDataset(x, x), batch_size=2)

    csv_logger = CSVLogger(log_dir=tmp_path / "logs", filename="gpt_train.csv")
    ppl_metric = Perplexity(loss_mode="lm")

    cfg = TrainConfig(
        max_epochs=2,
        lr=2e-3,
        optimizer="adamw",
        scheduler="none",
        checkpoint_dir=str(tmp_path),
        save_best=True,
        monitor="val_perplexity",
        mode="min",
    )
    trainer = Trainer(
        model,
        cfg,
        train_loader=loader,
        val_loader=loader,
        metrics=[ppl_metric],
        loggers=[csv_logger],
    )
    trainer.train()

    # Check best model directory exists and has model.safetensors and meta.json
    best_dir = tmp_path / "best"
    assert best_dir.exists()
    assert (best_dir / "model.safetensors").exists()
    assert (best_dir / "meta.json").exists()

    best_meta = json.loads((best_dir / "meta.json").read_text())
    assert "val_perplexity" in best_meta["metrics"]

    # Verify log CSV contains val_perplexity
    csv_file = tmp_path / "logs" / "gpt_train.csv"
    assert csv_file.exists()
    with open(csv_file, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert any("val_perplexity" in r for r in rows)


def test_integration_classifier_accuracy_and_best_model(tmp_path):
    """Train MLP classifier with Accuracy metric and best-model monitoring on val_accuracy (mode='max')."""
    torch.manual_seed(42)
    model = nn.Sequential(
        nn.Linear(8, 16),
        nn.ReLU(),
        nn.Linear(16, 2),
    )
    x = torch.randn(16, 8)
    y = (x[:, 0] > 0).long()
    loader = DataLoader(TensorDataset(x, y), batch_size=4)

    acc_metric = Accuracy()
    cfg = TrainConfig(
        max_epochs=2,
        lr=1e-2,
        optimizer="sgd",
        scheduler="none",
        loss="cross_entropy",
        checkpoint_dir=str(tmp_path),
        save_best=True,
        monitor="val_accuracy",
        mode="max",
    )
    trainer = Trainer(
        model,
        cfg,
        train_loader=loader,
        val_loader=loader,
        metrics=[acc_metric],
    )
    trainer.train()

    best_dir = tmp_path / "best"
    assert best_dir.exists()
    best_meta = json.loads((best_dir / "meta.json").read_text())
    assert "val_accuracy" in best_meta["metrics"]
    assert best_meta["metrics"]["val_accuracy"] >= 0.0
