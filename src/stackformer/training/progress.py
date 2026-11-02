"""Notebook-safe, rank-0 progress bar with plain text fallback."""

from __future__ import annotations

import sys
from typing import Any, Iterable, Iterator


class ProgressBar:
    """A progress bar that wraps tqdm.auto if available, falling back to plain stdout."""

    def __init__(
        self,
        iterable: Iterable[Any] | None = None,
        total: int | None = None,
        desc: str = "",
        leave: bool = True,
        enabled: bool = True,
    ) -> None:
        self.enabled = enabled
        self.total = total if total is not None else (len(iterable) if hasattr(iterable, "__len__") else None)
        self.desc = desc
        self.iterable = iterable
        self._tqdm_inst = None
        self._step = 0

        if self.enabled:
            try:
                from tqdm.auto import tqdm
                self._tqdm_inst = tqdm(
                    iterable=iterable,
                    total=self.total,
                    desc=desc,
                    leave=leave,
                    file=sys.stdout,
                )
            except ImportError:
                self._tqdm_inst = None

    def __iter__(self) -> Iterator[Any]:
        if not self.enabled:
            if self.iterable is not None:
                yield from self.iterable
            return

        if self._tqdm_inst is not None:
            yield from self._tqdm_inst
        else:
            if self.iterable is not None:
                for item in self.iterable:
                    self._step += 1
                    if self._step % max(1, (self.total or 100) // 10) == 0:
                        sys.stdout.write(f"\r{self.desc}: {self._step}/{self.total}")
                        sys.stdout.flush()
                    yield item
                sys.stdout.write("\n")
                sys.stdout.flush()

    def update(self, n: int = 1) -> None:
        """Update progress bar by n steps."""
        if not self.enabled:
            return
        if self._tqdm_inst is not None:
            self._tqdm_inst.update(n)
        else:
            self._step += n

    def set_postfix(self, **kwargs: Any) -> None:
        """Set postfix statistics (e.g., loss, lr)."""
        if not self.enabled:
            return
        if self._tqdm_inst is not None:
            formatted = {}
            for k, v in kwargs.items():
                if isinstance(v, float):
                    formatted[k] = f"{v:.4f}" if abs(v) < 1000 else f"{v:.2e}"
                else:
                    formatted[k] = v
            self._tqdm_inst.set_postfix(formatted, refresh=False)

    def close(self) -> None:
        """Close progress bar."""
        if self._tqdm_inst is not None:
            self._tqdm_inst.close()

