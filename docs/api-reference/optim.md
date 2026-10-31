# Optimizers & Loss Functions API Reference

This page documents the optimization factories, parameter grouping routines, and loss functions in `stackformer.optim`.

---

## Optimizer & Scheduler Factories

::: stackformer.optim.factories
    options:
      show_root_heading: true
      members:
        - create_optimizer
        - create_scheduler
        - get_parameter_groups

---

## Loss Functions

::: stackformer.optim.loss_fn
    options:
      show_root_heading: true
      members:
        - language_modeling_cross_entropy
        - classification_cross_entropy
        - binary_cross_entropy
        - segmentation_cross_entropy

