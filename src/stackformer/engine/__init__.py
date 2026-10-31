"""Engine package for model training execution, state management, and checkpointing.

Exposes:
    - Engine: Orchestrates single-device and distributed model execution
    - TrainingState: Tracks training step counters, epoch numbers, and metrics
    - Trainer: High-level model training loop wrapper
    - CheckpointManager: Saves and loads model state checkpoints
"""

from .checkpoint import CheckpointManager
from .engine import Engine
from .state import TrainingState
from .trainer import Trainer

__all__ = ["Engine", "TrainingState", "Trainer", "CheckpointManager"]

