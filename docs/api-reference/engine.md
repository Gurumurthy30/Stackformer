# Training Engine API Reference

This page documents the training execution, state management, and checkpointing components in `stackformer.engine` and `stackformer.amp`.

---

## High-Level Trainer

::: stackformer.engine.trainer
    options:
      show_root_heading: true
      members:
        - Trainer

---

## Runtime Step Engine

::: stackformer.engine.engine
    options:
      show_root_heading: true
      members:
        - Engine

---

## Training State Container

::: stackformer.engine.state
    options:
      show_root_heading: true
      members:
        - TrainingState

---

## Checkpoint Manager

::: stackformer.engine.checkpoint
    options:
      show_root_heading: true
      members:
        - CheckpointManager

---

## Automatic Mixed Precision (AMP)

::: stackformer.amp.scaler
    options:
      show_root_heading: true
      members:
        - AMPScaler
        - initialize_scaler
        - scale_loss
        - step_optimizer
        - update_scaler

