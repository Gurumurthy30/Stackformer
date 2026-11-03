"""Structured display manager with Rich support, plain fallback, and notebook detection."""

from __future__ import annotations

import logging
import re
import sys
from collections.abc import Iterable
from typing import Any

from stackformer.distributed.utils import is_rank_zero

from .progress import ProgressBar

logger = logging.getLogger(__name__)

ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;]*[a-zA-Z]")


def strip_ansi(text: str) -> str:
    """Remove ANSI escape sequences from text string."""
    return ANSI_ESCAPE_RE.sub("", text)


def is_in_notebook() -> bool:
    """Detect whether current execution environment is a Jupyter or IPython notebook."""
    try:
        from IPython import get_ipython  # type: ignore

        ip = get_ipython()
        if ip is not None and "IPKernelApp" in getattr(ip, "config", {}):
            return True
    except Exception:
        pass
    return bool("ipykernel" in sys.modules)


class DisplayManager:
    """Structured pretty output coordinator for Stackformer training."""

    def __init__(
        self,
        mode: str = "auto",
        verbosity: str = "normal",
        rank_zero_only: bool = True,
        file: Any | None = None,
    ) -> None:
        self.raw_mode = mode.lower().strip()
        self.verbosity = verbosity.lower().strip()
        self.rank_zero_only = rank_zero_only
        self.file = file or sys.stdout

        # Rank-0 check
        self.is_active = is_rank_zero() if rank_zero_only else True

        # Verbosity checks
        self.is_quiet = self.verbosity == "quiet"
        self.is_normal = self.verbosity == "normal"
        self.is_verbose = self.verbosity == "verbose"

        # TTY and notebook detection
        self.is_tty = hasattr(self.file, "isatty") and self.file.isatty()
        self.in_notebook = is_in_notebook()

        # Rich availability
        self._has_rich = False
        self._rich_console = None

        if self.raw_mode in ("auto", "rich"):
            try:
                import rich  # noqa: F401
                from rich.console import Console

                # When not a TTY and not in notebook, suppress ANSI colors
                use_color = self.is_tty or self.in_notebook
                self._rich_console = Console(
                    file=self.file,
                    force_terminal=use_color,
                    no_color=not use_color,
                    highlight=False,
                )
                self._has_rich = True
            except ImportError as e:
                if self.raw_mode == "rich":
                    raise ImportError(
                        "rich is required for DisplayManager(mode='rich'). "
                        "Install it with: pip install 'stackformer[train]' or pip install rich"
                    ) from e
                self._has_rich = False

        if self.raw_mode == "plain":
            self.mode = "plain"
        elif self.raw_mode == "rich":
            self.mode = "rich" if self._has_rich else "plain"
        else:  # auto
            self.mode = "rich" if self._has_rich else "plain"

    def write(self, text: str) -> None:
        """Write raw text directly to output file if rank-0 active and not quiet."""
        if not self.is_active or self.is_quiet:
            return
        if not (self.is_tty or self.in_notebook):
            text = strip_ansi(text)
        self.file.write(text)
        self.file.flush()

    def log(self, message: str) -> None:
        """Log message to output stream."""
        if not self.is_active or self.is_quiet:
            return
        if self.mode == "rich" and self._rich_console is not None:
            self._rich_console.print(message)
        else:
            self.write(message + "\n")

    def print_header(self, title: str, subtitle: str | None = None) -> None:
        """Render a styled header block."""
        if not self.is_active or self.is_quiet:
            return

        if self.mode == "rich" and self._rich_console is not None:
            from rich.panel import Panel

            content = f"[bold]{title}[/bold]"
            if subtitle:
                content += f"\n[dim]{subtitle}[/dim]"
            self._rich_console.print(Panel(content, expand=False, border_style="cyan"))
        else:
            sep = "=" * max(40, len(title) + 4)
            self.write(f"\n{sep}\n  {title}\n")
            if subtitle:
                self.write(f"  {subtitle}\n")
            self.write(f"{sep}\n")

    def print_config(
        self,
        config: Any,
        title: str = "Training Configuration",
    ) -> None:
        """Render configuration attributes in a two-column key-value table."""
        if not self.is_active or self.is_quiet:
            return

        items: list[tuple[str, str]] = []
        if hasattr(config, "to_dict"):
            cfg_dict = config.to_dict()
        elif isinstance(config, dict):
            cfg_dict = config
        else:
            cfg_dict = {
                k: getattr(config, k)
                for k in dir(config)
                if not k.startswith("_") and not callable(getattr(config, k))
            }

        # Show important training settings
        key_order = [
            "device",
            "parallel",
            "mixed_precision",
            "lr",
            "weight_decay",
            "optimizer",
            "scheduler",
            "warmup_steps",
            "grad_accum_steps",
            "max_grad_norm",
            "max_epochs",
            "max_steps",
            "seed",
        ]
        for k in key_order:
            if k in cfg_dict:
                items.append((k, str(cfg_dict[k])))
        for k, v in sorted(cfg_dict.items()):
            if k not in key_order and not isinstance(v, dict | list):
                items.append((k, str(v)))

        if self.mode == "rich" and self._rich_console is not None:
            from rich.table import Table

            table = Table(title=title, show_header=True, header_style="bold magenta")
            table.add_column("Parameter", style="cyan", no_wrap=True)
            table.add_column("Value", style="green")
            for param, val in items:
                table.add_row(param, val)
            self._rich_console.print(table)
        else:
            self.write(f"\n--- {title} ---\n")
            col_w = max(len(k) for k, _ in items) if items else 15
            for param, val in items:
                self.write(f"  {param:<{col_w}} : {val}\n")
            self.write("-----------------------------\n")

    def print_epoch_summary(
        self,
        epoch: int,
        max_epochs: int | None,
        train_loss: float,
        metrics: dict[str, Any] | None = None,
        val_metrics: dict[str, Any] | None = None,
        elapsed_sec: float | None = None,
    ) -> None:
        """Render an epoch completion summary."""
        if not self.is_active or self.is_quiet:
            return

        epoch_str = f"Epoch {epoch}/{max_epochs}" if max_epochs else f"Epoch {epoch}"
        time_str = f" [{elapsed_sec:.2f}s]" if elapsed_sec is not None else ""

        if self.mode == "rich" and self._rich_console is not None:
            from rich.table import Table

            title = f"{epoch_str} Complete{time_str}"
            table = Table(title=title, show_header=True, header_style="bold blue")
            table.add_column("Stage", style="bold")
            table.add_column("Loss", style="yellow")
            table.add_column("Metrics", style="green")

            m_str = " ".join(
                f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                for k, v in (metrics or {}).items()
            )
            table.add_row("Train", f"{train_loss:.4f}", m_str or "-")

            if val_metrics:
                v_loss = val_metrics.get("val_loss", "-")
                v_loss_str = f"{v_loss:.4f}" if isinstance(v_loss, float) else str(v_loss)
                v_str = " ".join(
                    f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                    for k, v in val_metrics.items()
                    if k != "val_loss"
                )
                table.add_row("Val", v_loss_str, v_str or "-")

            self._rich_console.print(table)
        else:
            self.write(f"\n>>> {epoch_str}{time_str} | Train Loss: {train_loss:.4f}")
            if metrics:
                m_str = " ".join(
                    f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                    for k, v in metrics.items()
                )
                self.write(f" | {m_str}")
            if val_metrics:
                v_loss = val_metrics.get("val_loss")
                if v_loss is not None:
                    self.write(f" | Val Loss: {v_loss:.4f}")
            self.write("\n")

    def print_eval_table(
        self,
        metrics: dict[str, Any],
        step: int | None = None,
        epoch: int | None = None,
    ) -> None:
        """Render evaluation metrics table."""
        if not self.is_active or self.is_quiet:
            return

        title_parts = ["Evaluation"]
        if epoch is not None:
            title_parts.append(f"Epoch {epoch}")
        if step is not None:
            title_parts.append(f"Step {step}")
        title = " - ".join(title_parts)

        if self.mode == "rich" and self._rich_console is not None:
            from rich.table import Table

            table = Table(title=title, show_header=True, header_style="bold green")
            table.add_column("Metric", style="cyan")
            table.add_column("Score", style="yellow")
            for k, v in sorted(metrics.items()):
                v_str = f"{v:.4f}" if isinstance(v, float) else str(v)
                table.add_row(k, v_str)
            self._rich_console.print(table)
        else:
            self.write(f"\n--- {title} ---\n")
            col_w = max(len(k) for k in metrics) if metrics else 12
            for k, v in sorted(metrics.items()):
                v_str = f"{v:.4f}" if isinstance(v, float) else str(v)
                self.write(f"  {k:<{col_w}} : {v_str}\n")
            self.write("-----------------------------\n")

    def print_step(self, step: int, loss: float, lr: float, **kwargs: Any) -> None:
        """Log individual optimizer step (verbose mode only)."""
        if not self.is_active or not self.is_verbose:
            return

        extra = " ".join(
            f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in kwargs.items()
        )
        msg = f"[Step {step}] Loss: {loss:.4f} | LR: {lr:.2e}"
        if extra:
            msg += f" | {extra}"

        if self.mode == "rich" and self._rich_console is not None:
            self._rich_console.print(msg)
        else:
            self.write(msg + "\n")

    def progress_bar(
        self,
        iterable: Iterable[Any] | None = None,
        total: int | None = None,
        desc: str = "",
        leave: bool = True,
    ) -> ProgressBar:
        """Instantiate an execution progress bar with rank and quiet mode respect."""
        enabled = self.is_active and not self.is_quiet
        return ProgressBar(
            iterable=iterable,
            total=total,
            desc=desc,
            leave=leave,
            enabled=enabled,
        )
