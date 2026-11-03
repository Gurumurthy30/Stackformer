"""License compliance and attribution verification tests for Stackformer."""

import importlib
import subprocess
import tarfile
import zipfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

DERIVED_FILES = [
    "src/stackformer/ops/__init__.py",
    "src/stackformer/ops/scatter_add.py",
    "src/stackformer/quantization/__init__.py",
    "src/stackformer/quantization/_fsdp_tensor.py",
    "src/stackformer/quantization/float8.py",
    "src/stackformer/quantization/utils.py",
    "src/stackformer/distributed/__init__.py",
    "src/stackformer/distributed/hardware.py",
    "src/stackformer/distributed/mesh.py",
    "src/stackformer/distributed/activation_checkpoint.py",
    "src/stackformer/distributed/compile.py",
    "src/stackformer/distributed/ddp.py",
    "src/stackformer/distributed/fsdp.py",
    "src/stackformer/distributed/utils.py",
    "src/stackformer/distributed/dcp.py",
]


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


def test_meta_bsd3_headers_in_derived_files():
    """Verify Meta BSD-3-Clause copyright header is intact in all derived files."""
    for rel_path in DERIVED_FILES:
        py_file = REPO_ROOT / rel_path
        assert py_file.is_file(), f"Derived file missing: {rel_path}"
        content = py_file.read_text(encoding="utf-8")
        first_500 = content[:500]
        assert "Copyright (c) Meta Platforms, Inc. and affiliates." in first_500, (
            f"Missing Meta copyright in {rel_path}"
        )
        assert "BSD-style license" in first_500, f"Missing BSD license notice in {rel_path}"


def test_notice_files_exist_and_contain_conditions_and_disclaimer():
    """Verify TORCHTITAN-BSD-3-Clause.txt and THIRD_PARTY_NOTICES.md exist and contain required clauses."""
    license_file = REPO_ROOT / "licenses" / "TORCHTITAN-BSD-3-Clause.txt"
    notices_file = REPO_ROOT / "THIRD_PARTY_NOTICES.md"

    assert license_file.is_file(), "Missing TORCHTITAN-BSD-3-Clause.txt in licenses/"
    assert notices_file.is_file(), "Missing THIRD_PARTY_NOTICES.md at repo root"

    license_text = license_file.read_text(encoding="utf-8")
    notices_text = notices_file.read_text(encoding="utf-8")

    # (c) Meta Platforms, Inc. and affiliates
    assert "Meta Platforms, Inc. and affiliates" in license_text
    assert "Meta Platforms, Inc. and affiliates" in notices_text

    # Three BSD conditions
    conditions = [
        "1. Redistributions of source code must retain the above copyright notice",
        "2. Redistributions in binary form must reproduce the above copyright notice",
        "3. Neither the name of the copyright holder nor the names of its contributors",
    ]
    for c in conditions:
        assert c in license_text, f"Missing condition '{c}' in licenses/TORCHTITAN-BSD-3-Clause.txt"
        assert c in notices_text, f"Missing condition '{c}' in THIRD_PARTY_NOTICES.md"

    # Disclaimer
    disclaimer_snippet = "THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS"
    assert disclaimer_snippet in license_text
    assert disclaimer_snippet in notices_text


def test_wheel_and_sdist_license_distribution(tmp_path):
    """Verify built wheel and sdist packages contain licenses/TORCHTITAN-BSD-3-Clause.txt and THIRD_PARTY_NOTICES.md."""
    try:
        import build
    except ImportError:
        pytest.skip("python -m build is not available in the test environment")

    # Build sdist and wheel with --no-isolation to use existing environment
    cmd = ["python", "-m", "build", "--no-isolation", "--outdir", str(tmp_path)]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO_ROOT)
    if result.returncode != 0:
        pytest.skip(f"Build failed, skipping package inspection: {result.stderr}")

    # Inspect wheel
    wheels = list(tmp_path.glob("*.whl"))
    assert len(wheels) == 1, "Expected exactly one wheel built"
    with zipfile.ZipFile(wheels[0]) as zf:
        namelist = zf.namelist()
        assert any("LICENSE" in name for name in namelist), (
            f"LICENSE not found in wheel: {namelist}"
        )
        assert any("TORCHTITAN-BSD-3-Clause.txt" in name for name in namelist), (
            f"TORCHTITAN-BSD-3-Clause.txt not found in wheel: {namelist}"
        )
        assert any("THIRD_PARTY_NOTICES" in name for name in namelist), (
            f"THIRD_PARTY_NOTICES not found in wheel: {namelist}"
        )

    # Inspect sdist
    sdists = list(tmp_path.glob("*.tar.gz"))
    assert len(sdists) == 1, "Expected exactly one sdist built"
    with tarfile.open(sdists[0]) as tf:
        namelist = tf.getnames()
        assert any("LICENSE" in name for name in namelist), (
            f"LICENSE not found in sdist: {namelist}"
        )
        assert any("TORCHTITAN-BSD-3-Clause.txt" in name for name in namelist), (
            f"TORCHTITAN-BSD-3-Clause.txt not found in sdist: {namelist}"
        )
        assert any("THIRD_PARTY_NOTICES" in name for name in namelist), (
            f"THIRD_PARTY_NOTICES not found in sdist: {namelist}"
        )
