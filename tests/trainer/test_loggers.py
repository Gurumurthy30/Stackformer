"""Unit tests for logging backends in stackformer.training.loggers."""

import csv
import json
from pathlib import Path
import pytest
import torch

from stackformer.training.loggers import (
    CSVLogger,
    ConsoleLogger,
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

    with open(csv_file, "r", encoding="utf-8") as f:
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

    with open(csv_file, "r", encoding="utf-8") as f:
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

    with open(jsonl_file, "r", encoding="utf-8") as f:
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

    with open(jsonl_file, "r", encoding="utf-8") as f:
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
