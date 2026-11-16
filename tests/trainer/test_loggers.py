"""Unit tests for logging backends in stackformer.training.loggers."""

import csv
import json
from pathlib import Path

import pytest
import torch

from stackformer.training.loggers import (
    ConsoleLogger,
    CSVLogger,
    JSONLLogger,
    TensorBoardLogger,
    WandbLogger,
)


def test_console_logger_smoke():
    logger = ConsoleLogger(verbose=True)
    logger.log_config({"lr": 1e-3, "epochs": 2})
    logger.log_metrics({"loss": 1.234, "acc": 0.85}, step=10)
    logger.finish()


def test_csv_logger_lifecycle_and_append(tmp_path):
    log_dir = tmp_path / "csv_logs"
    logger = CSVLogger(log_dir=log_dir, filename="run.csv", append=True)

    logger.log_metrics({"train_loss": 2.5, "lr": 1e-3}, step=1)
    logger.log_metrics({"train_loss": 2.1, "lr": 9e-4}, step=2)
    logger.finish()

    csv_file = log_dir / "run.csv"
    assert csv_file.exists()

    with open(csv_file, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == 2
    assert rows[0]["step"] == "1"
    assert float(rows[0]["train_loss"]) == 2.5
    assert rows[1]["step"] == "2"
    assert float(rows[1]["train_loss"]) == 2.1

    # Resume test: new logger in append mode adds step 3 without destroying step 1 & 2
    logger2 = CSVLogger(log_dir=log_dir, filename="run.csv", append=True)
    logger2.log_metrics({"train_loss": 1.8, "lr": 8e-4}, step=3)
    logger2.finish()

    with open(csv_file, encoding="utf-8") as f:
        rows_resumed = list(csv.DictReader(f))

    assert len(rows_resumed) == 3
    assert rows_resumed[2]["step"] == "3"
    assert float(rows_resumed[2]["train_loss"]) == 1.8


def test_jsonl_logger_lifecycle_and_append(tmp_path):
    log_dir = tmp_path / "jsonl_logs"
    logger = JSONLLogger(log_dir=log_dir, filename="run.jsonl", append=True)

    logger.log_config({"model": "gpt2", "embed_dim": 64})
    logger.log_metrics({"val_loss": 1.5, "val_acc": 0.75}, step=5)
    logger.finish()

    jsonl_file = log_dir / "run.jsonl"
    assert jsonl_file.exists()

    with open(jsonl_file, encoding="utf-8") as f:
        records = [json.loads(line) for line in f]

    assert len(records) == 2
    assert records[0]["type"] == "config"
    assert records[0]["config"]["model"] == "gpt2"
    assert records[1]["type"] == "metrics"
    assert records[1]["step"] == 5
    assert records[1]["val_loss"] == 1.5

    # Resume test: append new metric row
    logger2 = JSONLLogger(log_dir=log_dir, filename="run.jsonl", append=True)
    logger2.log_metrics({"val_loss": 1.2, "val_acc": 0.82}, step=10)
    logger2.finish()

    with open(jsonl_file, encoding="utf-8") as f:
        records_resumed = [json.loads(line) for line in f]

    assert len(records_resumed) == 3
    assert records_resumed[2]["step"] == 10
    assert records_resumed[2]["val_loss"] == 1.2


def test_tensorboard_logger_import_hint(monkeypatch, tmp_path):
    """Verify TensorBoardLogger gives helpful install hint when tensorboard package is missing."""
    import builtins

    orig_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        if "tensorboard" in name:
            raise ImportError("No module named 'tensorboard'")
        return orig_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", mock_import)

    with pytest.raises(ImportError, match="pip install tensorboard"):
        TensorBoardLogger(log_dir=tmp_path)


def test_wandb_logger_import_hint(monkeypatch):
    """Verify WandbLogger gives helpful install hint when wandb is missing."""
    import builtins

    orig_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        if name == "wandb":
            raise ImportError("No module named 'wandb'")
        return orig_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", mock_import)

    with pytest.raises(ImportError, match="pip install wandb"):
        WandbLogger(project="test-proj")


class InMemoryLogger(CSVLogger):
    def __init__(self):
        self.metrics_history: list[tuple[int, dict[str, float]]] = []
        self.flushed = False
        self.finished = False

    def log_metrics(self, metrics: dict[str, float], step: int) -> None:
        self.metrics_history.append((step, metrics))

    def flush(self) -> None:
        self.flushed = True

    def finish(self) -> None:
        self.finished = True


def test_epoch_metrics_logged_to_loggers():
    """Finding #37(a): dispatch_epoch_end logs epoch metrics to all registered loggers."""
    from torch.utils.data import DataLoader, TensorDataset

    from stackformer.config import ModelConfig
    from stackformer.models import GPT2
    from stackformer.training import TrainConfig, Trainer

    cfg = ModelConfig(
        vocab_size=32, embed_dim=16, num_layers=1, num_heads=2, seq_len=8, hidden_dim=32
    )
    model = GPT2(cfg)
    x = torch.randint(0, 32, (8, 6))
    loader = DataLoader(TensorDataset(x, x), batch_size=4)

    logger = InMemoryLogger()
    trainer = Trainer(
        model=model,
        train_loader=loader,
        config=TrainConfig(max_epochs=1, lr=1e-3),
        loggers=[logger],
    )
    trainer.train()

    epoch_metrics = [m for _, m in logger.metrics_history if "epoch/epoch_loss" in m]
    assert len(epoch_metrics) == 1, "Epoch loss must reach registered loggers"
    assert isinstance(epoch_metrics[0]["epoch/epoch_loss"], float)


def test_loggers_remain_active_after_train_until_close():
    """Finding #37(e): train() flushes loggers without closing them, allowing evaluate() to log, and close() is idempotent."""
    from torch.utils.data import DataLoader, TensorDataset

    from stackformer.config import ModelConfig
    from stackformer.models import GPT2
    from stackformer.training import TrainConfig, Trainer

    cfg = ModelConfig(
        vocab_size=32, embed_dim=16, num_layers=1, num_heads=2, seq_len=8, hidden_dim=32
    )
    model = GPT2(cfg)
    x = torch.randint(0, 32, (8, 6))
    loader = DataLoader(TensorDataset(x, x), batch_size=4)

    logger = InMemoryLogger()
    trainer = Trainer(
        model=model,
        train_loader=loader,
        val_loader=loader,
        config=TrainConfig(max_epochs=1, lr=1e-3),
        loggers=[logger],
    )
    trainer.train()

    assert logger.flushed is True
    assert logger.finished is False, "loggers must not be finalized in train() finally block"

    # evaluate() works after train() and logs metrics
    trainer.evaluate()
    val_logs = [m for _, m in logger.metrics_history if "val_loss" in m]
    assert len(val_logs) >= 1, (
        "evaluate() after train() must successfully record metrics to loggers"
    )

    # Trainer.close() closes loggers and is idempotent
    trainer.close()
    assert logger.finished is True
    trainer.close()  # Must not error on repeated close


def test_lr_monitor_does_not_double_log_lr():
    """Finding #37(d): LRMonitor enriches step metrics in place without writing duplicate logger rows."""
    from unittest.mock import MagicMock

    from stackformer.training.callbacks.lr_monitor import LRMonitor

    opt = MagicMock()
    opt.param_groups = [{"lr": 0.001}]
    trainer = MagicMock()
    trainer.optimizer = opt

    monitor = LRMonitor()
    metrics = {"loss": 1.0, "lr": 0.001}
    monitor.on_step_end(trainer, step=1, metrics=metrics)
    assert metrics == {"loss": 1.0, "lr": 0.001}

    # Step metrics without 'lr'
    metrics2 = {"loss": 0.9}
    monitor.on_step_end(trainer, step=2, metrics=metrics2)
    assert metrics2 == {"loss": 0.9, "lr": 0.001}

    # Multiple param groups
    opt.param_groups = [{"lr": 0.001}, {"lr": 0.0005}]
    metrics3 = {"loss": 0.8}
    monitor.on_step_end(trainer, step=3, metrics=metrics3)
    assert metrics3 == {"loss": 0.8, "lr": 0.001, "lr/group_0": 0.001, "lr/group_1": 0.0005}


def test_metrics_on_train_emits_warning():
    """Finding #37(b): Trainer emits a UserWarning when metrics_on_train=True."""
    from stackformer.config import ModelConfig
    from stackformer.models import GPT2
    from stackformer.training import TrainConfig, Trainer

    cfg = ModelConfig(
        vocab_size=32, embed_dim=16, num_layers=1, num_heads=2, seq_len=8, hidden_dim=32
    )
    model = GPT2(cfg)

    with pytest.warns(UserWarning, match="metrics_on_train is not implemented"):
        Trainer(model=model, config=TrainConfig(metrics_on_train=True, max_epochs=1))
