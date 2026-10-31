# Utilities API Reference

This page documents the helper functions and device utilities provided in `stackformer.utils`.

---

## Device & Hardware Utilities

::: stackformer.utils.device
    options:
      show_root_heading: true
      members:
        - get_device
        - print_device_info
        - clear_cuda_cache
        - synchronize
        - format_memory
        - move_to_device

---

## General Utilities

::: stackformer.utils.utils
    options:
      show_root_heading: true
      members:
        - seed_everything
        - count_parameters
        - ensure_dir
        - timestamp
        - is_main_process
        - print_once
        - get_rank
        - get_world_size

---

## Dimension Calculation

::: stackformer.utils.ff_utils
    options:
      show_root_heading: true
      members:
        - hidden_dim_helper

