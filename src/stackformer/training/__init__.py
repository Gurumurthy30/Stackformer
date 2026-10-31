"""Training loop primitives and evaluation utilities.

Exposes:
    - train_epoch: Run a single epoch training loop
    - eval_epoch: Run an epoch evaluation loop
    - predict_loop: Run inference prediction loop over a dataloader
    - train_loop: Flexible training loop helper
    - validation_loop: Flexible validation loop helper
"""

from .loops import eval_epoch, predict_loop, train_epoch, train_loop, validation_loop

__all__ = ["train_epoch", "eval_epoch", "predict_loop", "train_loop", "validation_loop"]

