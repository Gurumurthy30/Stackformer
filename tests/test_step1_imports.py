"""Test imports and Meta BSD-3 headers for Step 1 moved files."""

import importlib
from pathlib import Path

import pytest


def test_step1_module_imports():
    """Verify that all moved modules can be imported directly."""
    modules = [
        "stackformer.ops",
        "stackformer.ops.scatter_add",
        "stackformer.quantization",
        "stackformer.quantization._fsdp_tensor",
        "stackformer.quantization.float8",
        "stackformer.quantization.utils",
        "stackformer.distributed",
        "stackformer.distributed.mesh",
        "stackformer.distributed.hardware",
    ]
    for mod_name in modules:
        mod = importlib.import_module(mod_name)
        assert mod is not None, f"Failed to import {mod_name}"


def test_meta_bsd3_headers():
    """Verify Meta BSD-3-Clause copyright header is intact in all moved files."""
    repo_root = Path(__file__).resolve().parent.parent
    src_root = repo_root / "src" / "stackformer"

    dirs_to_check = ["ops", "quantization", "distributed"]
    checked_files = 0

    for d in dirs_to_check:
        dir_path = src_root / d
        assert dir_path.is_dir(), f"Expected directory {dir_path} to exist"
        for py_file in dir_path.glob("*.py"):
            checked_files += 1
            content = py_file.read_text(encoding="utf-8")
            first_500 = content[:500]
            assert "Copyright (c) Meta Platforms, Inc. and affiliates." in first_500, (
                f"Missing Meta copyright in {py_file}"
            )
            assert "BSD-style license" in first_500, f"Missing BSD license notice in {py_file}"

    assert checked_files >= 9, f"Expected at least 9 files checked, got {checked_files}"
