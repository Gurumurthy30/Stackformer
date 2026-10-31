"""Logging package providing unified logger backends and metric tracking.

Exposes:
    - Logger: Unified logging multiplexer for CSV, TensorBoard, and W&B backends
    - CSVLogger: CSV log output backend
    - TensorBoardLogger: TensorBoard summary writer backend
    - WandBLogger: Weights & Biases experiment logger backend
    - MetricTracker: Metric running average and step timer tracker
    - Metrics: PascalCase alias for MetricTracker
    - accuracy, perplexity, precision, recall, f1_score: Metric calculation functions
"""

from .csv_logger import CSVLogger
from .logger import Logger
from .metrics import MetricTracker, Metrics, accuracy, f1_score, metrics, perplexity, precision, recall
from .tensorboard_logger import TensorBoardLogger
from .wandb_logger import WandBLogger

__all__ = [
    "Logger",
    "CSVLogger",
    "TensorBoardLogger",
    "WandBLogger",
    "MetricTracker",
    "Metrics",
    "metrics",
    "accuracy",
    "perplexity",
    "precision",
    "recall",
    "f1_score",
]

