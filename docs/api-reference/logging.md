# Logging & Metrics API Reference

This page documents the metric tracking and logging multiplexers in `stackformer.logging`.

---

## Unified Multiplexer

::: stackformer.logging.logger
    options:
      show_root_heading: true
      members:
        - Logger

---

## Log Sinks

::: stackformer.logging.csv_logger
    options:
      show_root_heading: true
      members:
        - CSVLogger

::: stackformer.logging.tensorboard_logger
    options:
      show_root_heading: true
      members:
        - TensorBoardLogger

::: stackformer.logging.wandb_logger
    options:
      show_root_heading: true
      members:
        - WandBLogger

---

## Metrics & Statistical Accumulation

::: stackformer.logging.metrics
    options:
      show_root_heading: true
      members:
        - MetricTracker
        - accuracy
        - perplexity
        - distributed_mean

