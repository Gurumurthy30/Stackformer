"""Pyright baseline comparison utility for continuous integration.

Ensures that no new type errors are introduced beyond the established baseline.
Usage:
    python scripts/check_pyright.py [--update]
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

BASELINE_PATH = Path(".pyright-baseline.json")


def normalize_diagnostic(diag: dict) -> dict:
    """Normalize diagnostic entry for cross-platform matching."""
    file_path = diag.get("file", "")
    # Normalize to relative POSIX path
    try:
        rel_path = Path(file_path).resolve().relative_to(Path.cwd().resolve()).as_posix()
    except Exception:
        rel_path = file_path.replace("\\", "/")

    return {
        "file": rel_path,
        "rule": diag.get("rule", ""),
        "message": diag.get("message", "").strip(),
    }


def run_pyright() -> dict:
    """Execute pyright with JSON output."""
    cmd = [sys.executable, "-m", "pyright", "--outputjson"]
    proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
    try:
        return json.loads(proc.stdout)
    except Exception as e:
        print(f"Failed to parse pyright output: {e}\nRaw output:\n{proc.stdout}\n{proc.stderr}")
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Check pyright against baseline.")
    parser.add_argument("--update", action="store_true", help="Update the baseline file.")
    args = parser.parse_args()

    data = run_pyright()
    diagnostics = [normalize_diagnostic(d) for d in data.get("generalDiagnostics", [])]
    total_errors = len(diagnostics)

    if args.update or not BASELINE_PATH.exists():
        with open(BASELINE_PATH, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "total_errors": total_errors,
                    "diagnostics": diagnostics,
                },
                f,
                indent=2,
            )
        print(f"Updated pyright baseline with {total_errors} existing diagnostics at {BASELINE_PATH}.")
        sys.exit(0)

    with open(BASELINE_PATH, "r", encoding="utf-8") as f:
        baseline_data = json.load(f)

    baseline_diagnostics = baseline_data.get("diagnostics", [])
    baseline_set = {
        (d["file"], d["rule"], d["message"])
        for d in baseline_diagnostics
    }

    new_errors = []
    for d in diagnostics:
        key = (d["file"], d["rule"], d["message"])
        if key not in baseline_set:
            new_errors.append(d)

    print(f"Pyright check: {total_errors} errors found (baseline: {len(baseline_diagnostics)})")
    if new_errors:
        print(f"\n[FAIL] {len(new_errors)} new Pyright type errors introduced:")
        for err in new_errors:
            print(f"  {err['file']} [{err['rule']}]: {err['message']}")
        sys.exit(1)

    print(f"[SUCCESS] No new Pyright type errors introduced! (Baseline: {len(baseline_diagnostics)} -> Current: {total_errors})")
    sys.exit(0)


if __name__ == "__main__":
    main()

