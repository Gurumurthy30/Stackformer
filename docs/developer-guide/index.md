# Developer Guide

Welcome to the StackFormer Developer Guide. This section provides detailed documentation for contributors, maintainers, and researchers looking to understand the inner workings of the codebase, add new modules or architectures, run test suites, and follow project development practices.

---

## Guide Overview

| Guide | Description |
| :--- | :--- |
| [**Project Structure**](project-structure.md) | Comprehensive overview of the `src/` layout, package organization, assets, tests, and configuration files. |
| [**Software Architecture**](architecture.md) | In-depth breakdown of the modular composition engine, tensor invariants, and component interfaces. |
| [**Design Principles & Scope**](design-principles.md) | Architectural non-goals, constraints, PyTorch-idiomatic design patterns, and engineering philosophy. |
| [**Testing Suite**](testing.md) | How to run PyTest, understand device test marks (`@pytest.mark.gpu`), write unit and integration tests, and check coverage. |
| [**Contributing Guide**](contributing.md) | Contribution workflow, Git branch standards, coding conventions, docstring formatting, and pull request requirements. |
| [**Development Roadmap**](roadmap.md) | Current implementation status and planned future roadmap milestones. |

---

## Quick Setup for Development

To set up an editable development installation with all test and lint dependencies:

```bash
# Clone the repository
git clone https://github.com/stackformer-labs/Stackformer.git
cd Stackformer

# Install in editable mode with development dependencies
pip install -e ".[dev]"

# Verify the test suite passes
pytest -q
```

