# Stackformer Trainer Output Samples

This document provides reference captures of training output across Stackformer's display modes (`plain` and `rich`) and outlines visual criteria to eyeball when running training inside an interactive Jupyter notebook.

---

## 1. Plain Mode Output (`display="plain"`)

Plain mode is designed for headless servers, continuous integration (CI) runners, file redirection, or environments without terminal styling/ANSI support. It outputs plain ASCII formatting without ANSI escape sequences.

```text
========================================
  Stackformer Training
  Model: GPT2 | Device: CPU
========================================

--- Training Configuration ---
  device                   : auto
  parallel                 : none
  mixed_precision          : no
  lr                       : 0.001
  weight_decay             : 0.01
  optimizer                : adamw
  scheduler                : cosine
  warmup_steps             : 0
  grad_accum_steps         : 1
  max_grad_norm            : 1.0
  max_epochs               : 3
  max_steps                : None
  seed                     : 42
  activation_checkpointing : none
  async_checkpoint         : False
  checkpoint_dir           : ./checkpoints
  compile                  : False
  compile_mode             : None
  display                  : plain
  eval_every_n_epochs      : 1
  eval_every_n_steps       : None
  keep_last_n              : 3
  loss                     : cross_entropy
  loss_mode                : auto
  max_eval_steps           : None
  metrics_on_train         : False
  mode                     : min
  monitor                  : val_loss
  resume_from              : None
  save_best                : True
  save_best_state          : False
  save_every_n_epochs      : 1
  save_every_n_steps       : None
  save_last                : True
  verbosity                : normal
-----------------------------

Epoch 1/3: 100%|##########| 4/4 [00:00<00:00, 160.20it/s, loss=4.6031, lr=0.0005, step=4]
>>> Epoch 1/3 [4.25s] | Train Loss: 3.4521 | lr=0.0010 step=50 | Val Loss: 3.1205

Epoch 2/3: 100%|##########| 4/4 [00:00<00:00, 175.40it/s, loss=4.5610, lr=0.0000, step=8]
>>> Epoch 2/3 [3.98s] | Train Loss: 3.0112 | lr=0.0000 step=100 | Val Loss: 2.8940

Epoch 3/3: 100%|##########| 4/4 [00:00<00:00, 182.10it/s, loss=4.1200, lr=0.0000, step=12]
>>> Epoch 3/3 [3.85s] | Train Loss: 2.7410 | lr=0.0000 step=150 | Val Loss: 2.6510
```

---

## 2. Rich Mode Output (`display="rich"` or `display="auto"` in terminal/notebook)

When `rich` is installed and a terminal or notebook is detected, Stackformer renders clean Unicode panels, colored tables, and progress indicators:

```text
┌───────────────────────────┐
│ Stackformer Training      │
│ Model: GPT2 | Device: CPU │
└───────────────────────────┘
           Training Configuration           
┌──────────────────────────┬───────────────┐
│ Parameter                │ Value         │
├──────────────────────────┼───────────────┤
│ device                   │ auto          │
│ parallel                 │ none          │
│ mixed_precision          │ no            │
│ lr                       │ 0.001         │
│ weight_decay             │ 0.01          │
│ optimizer                │ adamw         │
│ scheduler                │ cosine        │
│ warmup_steps             │ 0             │
│ grad_accum_steps         │ 1             │
│ max_grad_norm            │ 1.0           │
│ max_epochs               │ 3             │
│ max_steps                │ None          │
│ seed                     │ 42            │
│ activation_checkpointing │ none          │
│ async_checkpoint         │ False         │
│ checkpoint_dir           │ ./checkpoints │
│ compile                  │ False         │
│ compile_mode             │ None          │
│ display                  │ auto          │
│ eval_every_n_epochs      │ 1             │
│ eval_every_n_steps       │ None          │
│ keep_last_n              │ 3             │
│ loss                     │ cross_entropy │
│ loss_mode                │ auto          │
│ max_eval_steps           │ None          │
│ metrics_on_train         │ False         │
│ mode                     │ min           │
│ monitor                  │ val_loss      │
│ resume_from              │ None          │
│ save_best                │ True          │
│ save_best_state          │ False         │
│ save_every_n_epochs      │ 1             │
│ save_every_n_steps       │ None          │
│ save_last                │ True          │
│ verbosity                │ normal        │
└──────────────────────────┴───────────────┘

Epoch 1/3 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 4/4 [loss=4.6031, lr=0.0005, step=4]
      Epoch 1/3 Complete [4.25s]      
┌───────┬────────┬───────────────────┐
│ Stage │ Loss   │ Metrics           │
├───────┼────────┼───────────────────┤
│ Train │ 3.4521 │ lr=0.0010 step=50 │
│ Val   │ 3.1205 │ -                 │
└───────┴────────┴───────────────────┘
```

---

## 3. What to Eyeball in a Real Jupyter Notebook

When testing or demonstrating Stackformer within a JupyterLab, VS Code Jupyter Notebook, or Google Colab session, inspect the following elements:

### 1. In-Place Progress Bar Updates (No Newline Spam)
- **Check:** As training batches execute, does the progress bar update in-place in its cell output area?
- **Failure pattern:** Each batch printing a brand-new line (e.g. 50 separate `1%`, `2%`, `3%` lines scrolling off the screen).
- **Expected:** A single clean bar per epoch that smoothly fills from 0% to 100%.

### 2. Live Dynamic Metrics Postfix
- **Check:** Look at the right-hand postfix string of the progress bar during training.
- **Expected fields:**
  - `loss`: Current smoothed training loss (should decrease or fluctuate realistically).
  - `lr`: Current scheduled learning rate (e.g., following cosine decay or warmup).
  - `step`: Current monotonic global step counter.
  - `it/s` or throughput speed.

### 3. Training Configuration Table Formatting
- **Check:** Before step 1 begins, the `DisplayManager` renders a structured configuration table.
- **Expected:** Two neatly aligned columns (`Parameter` and `Value`). In Jupyter with Rich enabled, it renders with crisp borders and colored headers. In plain mode, clean tab-aligned ASCII.

### 4. Epoch Summary and Evaluation Tables
- **Check:** At the end of each epoch or validation interval:
  - An epoch summary block appears with elapsed time in seconds.
  - A comparison table displays `Train` vs `Val` loss and computed evaluation metrics.
  - The previous epoch's progress bar remains cleanly frozen at 100% or transitions smoothly into the next epoch.

### 5. Checkpointing Log Signals
- **Check:** When a new best evaluation metric is reached (or at scheduled save intervals):
  - Checkpoint notice is logged (e.g., `Saved best model checkpoint to checkpoints/step_...`).
  - The pointer file `checkpoints/latest` is updated to record the latest checkpoint directory name.

### 6. Clean Final State
- **Check:** When `trainer.train()` finishes:
  - The cell returns a populated `TrainState` object (`step`, `epoch`, `best_metric`, `elapsed_time`).
  - No background worker threads or zombie sub-processes remain active.

