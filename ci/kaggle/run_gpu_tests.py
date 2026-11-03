"""Kaggle GPU CI test runner script.

Clones the target repository, installs dependencies, and runs runtime validation tests.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def run(cmd: list[str], cwd: Path | None = None) -> None:
    """Execute shell command list and check return code.

    Args:
        cmd (list[str]): Command tokens list.
        cwd (Path | None, default=None): Working directory path.
    """
    printable = " ".join(cmd)
    print(f"\n[cmd] {printable}", flush=True)
    subprocess.run(cmd, check=True, cwd=cwd)


def run_with_fallback(primary: list[str], fallback: list[str], cwd: Path) -> None:
    """Attempt primary command and fallback to alternative command on error.

    Args:
        primary (list[str]): Primary command token list.
        fallback (list[str]): Fallback command token list.
        cwd (Path): Working directory path.
    """
    try:
        run(primary, cwd=cwd)
    except subprocess.CalledProcessError:
        print("[gpu-ci] Primary dependency install failed; trying fallback.", flush=True)
        run(fallback, cwd=cwd)


REPO = "__STACKFORMER_REPO__"
REF = "__STACKFORMER_REF__"
WORKDIR = Path("/kaggle/working")
SRC = WORKDIR / "Stackformer"


def main() -> None:
    """Execute lightweight GPU validation workflow."""
    print("[gpu-ci] ===== Stackformer lightweight GPU validation started =====", flush=True)
    run([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
    run(["git", "clone", "--depth", "1", REPO, str(SRC)])
    run(["git", "fetch", "--depth", "1", "origin", REF], cwd=SRC)
    run(["git", "checkout", REF], cwd=SRC)

    run_with_fallback(
        [sys.executable, "-m", "pip", "install", "-e", ".[dev]"],
        [sys.executable, "-m", "pip", "install", "-e", "."],
        cwd=SRC,
    )

    print("[gpu-ci] Running CPU + minimal GPU tests.", flush=True)
    run([sys.executable, "ci/kaggle/test_runtime.py"], cwd=SRC)
    print("[gpu-ci] ===== Stackformer lightweight GPU validation succeeded =====", flush=True)


if __name__ == "__main__":
    main()
