"""Verify relative markdown links across the repository."""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Matches [label](target)
MD_LINK_PATTERN = re.compile(r"\[(?:[^\]]*)\]\(([^)]+)\)")
# Matches <a href="target">
HTML_LINK_PATTERN = re.compile(r'<a\s+(?:[^>]*?\s+)?href=["\']([^"\']+)["\']')


def is_external(target: str) -> bool:
    """Return True if link target is external, file URI, or anchor-only."""
    t = target.strip()
    return (
        t.startswith("http://")
        or t.startswith("https://")
        or t.startswith("mailto:")
        or t.startswith("ftp://")
        or t.startswith("file://")
        or t.startswith("#")
    )


def target_exists(file_path: Path, target: str) -> bool:
    """Check if target path exists relative to file or relative to repo root."""
    clean_target = target.split("#")[0].strip()
    if not clean_target:
        return True

    # Normalize trailing slash e.g. getting-started/quickstart/ -> getting-started/quickstart
    trimmed = clean_target.rstrip("/\\")

    candidates = [
        file_path.parent / clean_target,
        REPO_ROOT / clean_target.lstrip("/\\"),
        file_path.parent / trimmed,
        REPO_ROOT / trimmed.lstrip("/\\"),
    ]

    # If it's a directory link, also test .md or /index.md
    if clean_target.endswith("/") or clean_target.endswith("\\"):
        candidates.append(file_path.parent / f"{trimmed}.md")
        candidates.append(REPO_ROOT / f"{trimmed.lstrip('/\\')}.md")
        candidates.append(file_path.parent / trimmed / "index.md")
        candidates.append(REPO_ROOT / trimmed.lstrip("/\\") / "index.md")

    return any(c.exists() for c in candidates)


def check_file_links(file_path: Path) -> list[str]:
    """Check all markdown links within a single file. Return list of error messages."""
    errors = []
    try:
        content = file_path.read_text(encoding="utf-8")
    except Exception as exc:
        return [f"Could not read {file_path}: {exc}"]

    targets = MD_LINK_PATTERN.findall(content) + HTML_LINK_PATTERN.findall(content)

    for raw_target in targets:
        target = raw_target.strip().split()[0]
        if is_external(target):
            continue

        if not target_exists(file_path, target):
            errors.append(f"{file_path.relative_to(REPO_ROOT)}: broken link '{target}'")

    return errors


def main() -> int:
    """Scan all markdown files and report broken links."""
    all_errors = []
    md_files = []

    for root, _dirs, files in os.walk(REPO_ROOT):
        # Skip git, venv, build artifacts, site, and reviews
        if any(
            ignored in root
            for ignored in [".git", ".venv", "site", "__pycache__", ".pytest_cache", "reviews"]
        ):
            continue
        for f in files:
            if f.endswith(".md"):
                md_files.append(Path(root) / f)

    for md_file in sorted(md_files):
        file_errors = check_file_links(md_file)
        all_errors.extend(file_errors)

    if all_errors:
        print(f"Found {len(all_errors)} broken relative link(s):")
        for err in all_errors:
            print(f"  - {err}")
        return 1

    print(f"All relative links in {len(md_files)} markdown files resolved successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
