# Testing Guide

StackFormer maintains a comprehensive automated test suite powered by `pytest` located in the `tests/` directory. Tests cover unit correctness, tensor output shapes, gradient flow, device placement, distributed synchronization, and architectural regressions.

---

## Running the Test Suite

### Full Test Suite (CPU)

Run all tests excluding GPU-only hardware tests:

```bash
pytest -v
```

Or for compact summary output:

```bash
pytest -q
```

### Running Specific Test Categories

Target individual subsystems by specifying their folder:

```bash
# Core modules (BlockConfig, attention, FFN, norm, embeddings)
pytest tests/modules/

# Pre-assembled language models (GPT, LLaMA, Gemma, BERT, RoBERTa, Transformer)
pytest tests/models/

# Vision models (ViT, SegFormer)
pytest tests/vision/

# Training engine, checkpointing, and state management
pytest tests/trainer/

# Utility functions and device helpers
pytest tests/utils/
```

### Running GPU-Specific Tests

Tests requiring active CUDA hardware are decorated with `@pytest.mark.gpu`. On CPU-only environments, these tests are automatically skipped:

```bash
# Run only GPU-marked tests on a CUDA machine
pytest -m gpu

# Run all tests except GPU tests
pytest -m "not gpu"
```

---

## Testing Utilities & Fixtures

The test suite uses shared fixtures defined in `tests/conftest.py` and helper functions in `tests/_test_utils.py`:

### `torch_device` Fixture

Automatically injects the active compute device (`"cpu"` or `"cuda"`) into test cases:

```python
def test_forward_pass(torch_device):
    model = GPT2(vocab_size=100, num_layers=2, embed_dim=64, num_heads=2, seq_len=32).to(torch_device)
    x = torch.randint(0, 100, (2, 16), device=torch_device)
    out = model(x)
    assert out.shape == (2, 16, 100)
    assert torch.isfinite(out).all()
```

### Numerical Stability Verification

All test cases verify that outputs remain finite (no `inf` or `nan` values):

```python
assert torch.isfinite(output).all(), "Model output contains NaN or Inf values"
```

---

## Writing New Tests

When contributing a new module, model, or feature:
1. **Locate the Target Test Directory**: Add your test under the matching subdirectory in `tests/` (e.g. `tests/modules/`, `tests/models/`).
2. **Deterministic Inputs**: Fix random seeds or use the `torch_device` fixture.
3. **Verify Forward & Backward**: Ensure gradients flow back to all learnable parameters:
   ```python
   loss = output.sum()
   loss.backward()
   for name, param in model.named_parameters():
       if param.requires_grad:
           assert param.grad is not None, f"Gradient missing for {name}"
   ```
4. **Test Output Shapes**: Assert tensor dimensions match expectations for varying sequence lengths and batch sizes.

