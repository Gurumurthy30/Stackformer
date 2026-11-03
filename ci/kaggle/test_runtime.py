"""Lightweight CPU and CUDA runtime validation tests for CI environments.

Verifies basic forward, backward gradient computation, and loss finiteness on CPU and GPU.
"""

from __future__ import annotations

import sys
import traceback

import torch


def _log(msg: str) -> None:
    """Print formatted CI log line.

    Args:
        msg (str): Log message string.
    """
    print(f"[gpu-ci] {msg}", flush=True)


def assert_true(condition: bool, message: str) -> None:
    """Assert boolean condition or raise AssertionError with descriptive message.

    Args:
        condition (bool): Expression to evaluate.
        message (str): Assertion failure message.
    """
    if not condition:
        raise AssertionError(message)


def run_cpu_test() -> None:
    """Run minimal CPU linear model forward/backward pass test."""
    _log("CPU test: tiny linear model forward/backward start")
    device = torch.device("cpu")
    _log(f"CPU test device: {device}")

    torch.manual_seed(0)
    model = torch.nn.Sequential(
        torch.nn.Linear(8, 16),
        torch.nn.ReLU(),
        torch.nn.Linear(16, 4),
    ).to(device)

    x = torch.randn(6, 8, device=device)
    y = model(x)
    loss = y.pow(2).mean()
    loss.backward()

    assert_true(y.shape == (6, 4), "CPU forward shape check failed")
    assert_true(model[0].weight.grad is not None, "CPU backward grad missing")
    assert_true(torch.isfinite(loss).item(), "CPU loss is not finite")
    _log("CPU test: PASS")


def run_gpu_test() -> None:
    """Run minimal CUDA GPU linear model forward/backward pass test."""
    _log("GPU test: minimal CUDA forward/backward start")
    assert_true(torch.cuda.is_available(), "CUDA unavailable")

    device = torch.device("cuda")
    _log(f"GPU test device: {device}")
    _log(f"GPU name: {torch.cuda.get_device_name(0)}")

    torch.manual_seed(0)
    model = torch.nn.Sequential(
        torch.nn.Linear(8, 16),
        torch.nn.ReLU(),
        torch.nn.Linear(16, 4),
    ).to(device)

    x = torch.randn(6, 8, device=device)
    y = model(x)
    loss = y.pow(2).mean()
    loss.backward()

    assert_true(next(model.parameters()).is_cuda, "Model did not move to CUDA")
    assert_true(y.is_cuda, "Forward output is not CUDA tensor")
    assert_true(model[0].weight.grad is not None, "GPU backward grad missing")
    assert_true(torch.isfinite(loss).item(), "GPU loss is not finite")
    _log("GPU test: PASS")


def main() -> None:
    """Execute CPU and GPU test functions."""
    _log(f"Python: {sys.version.split()[0]}")
    _log(f"PyTorch: {torch.__version__}")
    _log(f"CUDA available: {torch.cuda.is_available()}")

    failures: list[str] = []

    for name, fn in (("cpu", run_cpu_test), ("gpu", run_gpu_test)):
        try:
            fn()
        except Exception as exc:
            failures.append(f"{name}: {exc}")
            traceback.print_exc()

    if failures:
        _log("Test results: FAIL")
        for failure in failures:
            _log(f" - {failure}")
        raise SystemExit(1)

    _log("Test results: PASS")


if __name__ == "__main__":
    main()
