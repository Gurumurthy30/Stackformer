"""File-based metrics loggers: CSVLogger and JSONLLogger with append support."""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from stackformer.training.loggers.base import Logger, is_rank_zero


class CSVLogger(Logger):
    """Logs training metrics to a CSV spreadsheet file.

    Maintains append mode across training resumes so prior metrics are preserved.

    Args:
        log_dir: Directory where the CSV file is saved (default: "./checkpoints/logs").
        filename: Name of the CSV file (default: "metrics.csv").
        append: Whether to append to existing file on resume (default: True).
    """

    def __init__(
        self,
        log_dir: str | Path | None = None,
        filename: str = "metrics.csv",
        append: bool = True,
    ) -> None:
        self.log_dir = Path(log_dir or "./checkpoints/logs").resolve()
        self.filename = filename
        self.append = append
        self.file_path = self.log_dir / self.filename
        self.fieldnames: list[str] = []
        self._file = None
        self._writer = None

    def _ensure_fields(self, field_candidates: list[str]) -> None:
        new_fields = [f for f in field_candidates if f not in self.fieldnames]
        if self._file is not None:
            if not new_fields:
                return
            # Close file and rewrite with expanded headers
            self._file.flush()
            self._file.close()
            self._file = None
            self._writer = None

            existing_rows = []
            if self.file_path.exists() and self.file_path.stat().st_size > 0:
                with open(self.file_path, encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    existing_rows = list(reader)

            self.fieldnames.extend(new_fields)
            with open(self.file_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=self.fieldnames, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(existing_rows)

            self._file = open(self.file_path, mode="a", newline="", encoding="utf-8")
            self._writer = csv.DictWriter(self._file, fieldnames=self.fieldnames, extrasaction="ignore")
            return

        self.log_dir.mkdir(parents=True, exist_ok=True)
        file_exists = self.file_path.exists() and self.file_path.stat().st_size > 0

        if file_exists and self.append:
            with open(self.file_path, encoding="utf-8") as f:
                reader = csv.reader(f)
                header = next(reader, None)
                if header:
                    self.fieldnames = list(header)

        for field in ["step", "timestamp", *field_candidates]:
            if field not in self.fieldnames:
                self.fieldnames.append(field)

        mode = "a" if (self.append and file_exists) else "w"
        self._file = open(self.file_path, mode=mode, newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._file, fieldnames=self.fieldnames, extrasaction="ignore")
        if mode == "w":
            self._writer.writeheader()
            self._file.flush()

    def log_metrics(self, metrics: dict[str, float], step: int) -> None:
        if not is_rank_zero():
            return
        keys = list(metrics.keys())
        self._ensure_fields(keys)

        row = {
            "step": step,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **metrics,
        }
        self._writer.writerow(row)
        self._file.flush()

    def finish(self) -> None:
        if self._file is not None and not self._file.closed:
            self._file.flush()
            self._file.close()
            self._file = None
            self._writer = None


class JSONLLogger(Logger):
    """Logs training metrics as JSON Lines with zero external dependencies.

    Args:
        log_dir: Directory where the JSONL file is saved (default: "./checkpoints/logs").
        filename: Name of the JSONL file (default: "metrics.jsonl").
        append: Whether to append to existing file on resume (default: True).
    """

    def __init__(
        self,
        log_dir: str | Path | None = None,
        filename: str = "metrics.jsonl",
        append: bool = True,
    ) -> None:
        self.log_dir = Path(log_dir or "./checkpoints/logs").resolve()
        self.filename = filename
        self.append = append
        self.file_path = self.log_dir / self.filename
        self._file = None

    def _ensure_open(self) -> None:
        if self._file is not None:
            return
        self.log_dir.mkdir(parents=True, exist_ok=True)
        file_exists = self.file_path.exists() and self.file_path.stat().st_size > 0
        mode = "a" if (self.append and file_exists) else "w"
        self._file = open(self.file_path, mode=mode, encoding="utf-8")

    def log_config(self, config_dict: dict[str, Any]) -> None:
        if not is_rank_zero():
            return
        self._ensure_open()
        record = {
            "type": "config",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "config": config_dict,
        }
        self._file.write(json.dumps(record) + "\n")
        self._file.flush()

    def log_metrics(self, metrics: dict[str, float], step: int) -> None:
        if not is_rank_zero():
            return
        self._ensure_open()
        record = {
            "type": "metrics",
            "step": step,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **metrics,
        }
        self._file.write(json.dumps(record) + "\n")
        self._file.flush()

    def finish(self) -> None:
        if self._file is not None and not self._file.closed:
            self._file.flush()
            self._file.close()
            self._file = None
