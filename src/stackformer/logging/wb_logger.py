"""Backward-compatible module alias for Weights & Biases logger (`WandBLogger`).

Exposes:
    - WandBLogger: Weights & Biases logger backend
"""

from .wandb_logger import WandBLogger

__all__ = ["WandBLogger"]

