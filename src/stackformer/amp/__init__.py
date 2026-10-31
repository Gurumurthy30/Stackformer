"""Automatic mixed precision (AMP) scaling and autocasting utilities.

Exposes:
    - AMPScaler: Wrapper for PyTorch GradScaler and autocast context
    - initialize_scaler: Factory function for constructing AMPScaler
    - scale_loss: Helper to scale loss tensor with optional scaler
    - step_optimizer: Helper to step optimizer with optional scaler
    - update_scaler: Helper to update scaler dynamic factor
"""

from .scaler import AMPScaler, initialize_scaler, scale_loss, step_optimizer, update_scaler

__all__ = ["AMPScaler", "initialize_scaler", "scale_loss", "step_optimizer", "update_scaler"]

