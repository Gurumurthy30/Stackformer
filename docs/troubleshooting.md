# Troubleshooting & Common Issues

This guide documents common errors, edge cases, and runtime pitfalls encountered when configuring, training, or running inference with StackFormer, along with precise diagnostic steps and solutions.

---

## 1. Dimensionality & Divisibility Errors

### `ValueError: embed_dim must be divisible by num_heads`

**Symptom**:
```text
ValueError: embed_dim (500) must be divisible by num_heads (8)
```

**Cause**:
Multi-head attention projects hidden states into $H$ distinct subspace heads of dimension $d_k = \frac{\text{embed\_dim}}{H}$. If `embed_dim` is not evenly divisible by `num_heads`, attention head projections cannot be partitioned cleanly.

**Solution**:
Ensure `embed_dim % num_heads == 0`.
```python
# Incorrect
cfg = BlockConfig(embed_dim=500, num_heads=8)

# Correct (512 is divisible by 8 -> head_dim = 64)
cfg = BlockConfig(embed_dim=512, num_heads=8)
```

---

### `ValueError: num_heads must be divisible by num_kv_heads for gqa`

**Symptom**:
```text
ValueError: num_heads (8) must be divisible by num_kv_heads (3) for gqa
```

**Cause**:
Grouped-Query Attention (GQA) groups query heads into clusters that share a single Key/Value head. The group size ratio $\frac{H_Q}{H_{KV}}$ must be an integer.

**Solution**:
Ensure `num_heads % num_kv_heads == 0`. Common ratios are 2, 4, or 8 queries per KV head:
```python
# Correct configurations:
# 8 query heads, 2 KV heads (4 queries per KV head)
cfg = BlockConfig(embed_dim=512, num_heads=8, attention="gqa", num_kv_heads=2)

# 8 query heads, 4 KV heads (2 queries per KV head)
cfg = BlockConfig(embed_dim=512, num_heads=8, attention="gqa", num_kv_heads=4)

# 8 query heads, 1 KV head (Multi-Query Attention equivalent)
cfg = BlockConfig(embed_dim=512, num_heads=8, attention="mqa", num_kv_heads=1)
```

---

## 2. Sequence Length & Context Capacity

### `ValueError: KV cache capacity exceeded: ... > cache length`

**Symptom**:
```text
ValueError: Prompt length 256 exceeds KV cache capacity 128. Truncate the prompt or rebuild the model with a larger kv_seq_len.
```

**Cause**:
When using models with stateful pre-allocated KV caches (e.g. `Llama2` with `batch_size` and `kv_seq_len`), the cache buffer is fixed at initialization to eliminate dynamic CUDA memory reallocations. If the prompt length or total decoded sequence exceeds `kv_seq_len`, the buffer overflows.

**Solution**:
1. Increase `kv_seq_len` when constructing the model:
   ```python
   model = Llama2(vocab_size=32000, num_layers=4, embed_dim=512, num_query_heads=8, num_kv_heads=2, seq_len=512)
   ```
2. Or truncate the input prompt before calling `text_generate()`.

---

## 3. Triton & CUDA Warnings on Windows

### `UserWarning: Failed to find CUDA. Install CUDA Toolkit...`

**Symptom**:
```text
UserWarning: Failed to find CUDA. Install CUDA Toolkit (https://developer.nvidia.com/cuda-downloads) or set CUDA_HOME / CUDA_PATH environment variable.
```

**Cause**:
This warning comes from Triton's Windows compatibility utilities (`triton.windows_utils`) when running in a pure CPU environment or when CUDA development headers are not on the system path.

**Impact on StackFormer**:
None. StackFormer's core layers, models, and training engine execute natively on standard PyTorch `nn.Module` and `F.scaled_dot_product_attention`. The fallback routines function seamlessly on CPU.

**Solution**:
If you are developing on a CPU-only machine, this warning is informational and can be ignored. If you are on an NVIDIA GPU machine, install CUDA Toolkit 12.x and set:
```bash
set CUDA_HOME="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4"
```

---

## 4. Training Engine & Checkpointing Issues

### Checkpoint Loading Mismatches with SafeTensors

**Symptom**:
```text
KeyError: "Missing key 'layers.0.norm1.weight' in state_dict"
```

**Cause**:
The saved model configuration does not match the architecture instantiated in code (e.g. attempting to load a `pre_norm=True` checkpoint into a `pre_norm=False` model, or loading a checkpoint saved with different `embed_dim` or `num_layers`).

**Solution**:
Instantiate the exact model architecture before restoring weights:
```python
from stackformer.engine import CheckpointManager

checkpoint_mgr = CheckpointManager(output_dir="checkpoints")

# Load model weights
checkpoint_mgr.load(model, checkpoint_name="checkpoint_epoch_1")
```

---

### Out of Memory (CUDA OOM) During Training

**Symptom**:
```text
torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate 1.2 GiB...
```

**Solutions**:
1. **Enable Automatic Mixed Precision (AMP)**:
   ```python
   with torch.autocast(device_type="cuda", dtype=torch.float16):
       outputs = model(inputs)
   ```
2. **Accumulate Gradients**:
   Reduce `batch_size` in your `DataLoader` by 2x or 4x and accumulate gradients over micro-steps before calling `optimizer.step()`.

3. **Use Grouped-Query Attention (GQA)**:
   GQA reduces KV cache memory footprint by 4x to 8x compared to standard Multi-Head Attention.

---

## 5. Distributed Data Parallel (DDP) Issues

### Hang During DDP Process Group Initialization

**Symptom**:
Script hangs indefinitely at `init_distributed()` or `wrap_model_ddp()`.

**Cause**:
- Rank 0 did not launch, or a network firewall blocked inter-process communication on `MASTER_PORT`.
- Using `nccl` backend on Windows (which only supports `gloo`).

**Solution**:
1. Always launch multi-process jobs via `torchrun`:
   ```bash
   torchrun --nproc_per_node=2 examples/train_ddp.py
   ```
2. On CPU or Windows environments, specify `backend="gloo"` when initializing process groups.
3. Always destroy process groups on termination via `torch.distributed.destroy_process_group()`.

