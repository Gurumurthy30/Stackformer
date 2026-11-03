"""Lifecycle callbacks and events subsystem for Stackformer Trainer."""

from __future__ import annotations

from stackformer.training.callbacks.base import Callback
from stackformer.training.callbacks.early_stopping import EarlyStopping
from stackformer.training.callbacks.lr_monitor import LRMonitor

__all__ = [
    "Callback",
    "EarlyStopping",
    "LRMonitor",
]
