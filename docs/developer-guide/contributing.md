# Contributing to StackFormer

Thank you for your interest in contributing to StackFormer! We welcome community contributions, including bug reports, new Transformer modules, performance optimizations, and documentation improvements.

---

## Development Workflow

### 1. Fork & Clone

Fork the repository on GitHub and clone your fork locally:

```bash
git clone https://github.com/<your-username>/Stackformer.git
cd Stackformer
```

### 2. Setup Development Environment

Install StackFormer in editable mode with all development, linting, and testing tools:

```bash
pip install -e ".[dev]"
```

### 3. Create a Feature Branch

Create a dedicated branch for your work:

```bash
git checkout -b feature/my-new-attention-module
```

---

## Coding Standards

StackFormer adheres to strict code quality standards:

1. **Python 3.10+ Support**: Use modern Python syntax and features (e.g. `from __future__ import annotations`, pipe type unions `X | Y`).
2. **Type Annotations**: All public functions, classes, and methods must include explicit type hints.
3. **Docstring Format**: Write comprehensive Google-style docstrings for all classes and functions, including:
   - Brief one-sentence explanation.
   - `Constructor args` or `Args` section with types and descriptions.
   - `Returns` section with tensor shapes.
   - Self-contained `Example` doctest.
4. **PyTorch Idioms**: Use standard `torch.nn.Module` patterns. Do not hide tensor shapes; document inputs as `(B, T, C)` or `(B, C, H, W)`.
5. **Preserve Compatibility**: Maintain backward-compatible aliases when renaming symbols.

---

## Testing Requirements

Before submitting a Pull Request, ensure that:
1. **The test suite passes completely**:
   ```bash
   pytest -q
   ```
2. **New functionality includes tests**: Add corresponding unit tests under `tests/` verifying output shapes, numerical finiteness, and gradient propagation.
3. **Documentation builds without warnings**:
   ```bash
   mkdocs build --strict
   ```

---

## Submitting a Pull Request

1. Push your branch to your GitHub fork:
   ```bash
   git push -u origin feature/my-new-attention-module
   ```
2. Open a Pull Request against the `main` branch.
3. Describe the change, reference any related issues, and provide sample output or benchmark numbers if applicable.
4. Maintainers will review your PR and run CI validation.

