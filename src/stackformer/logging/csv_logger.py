"""CSV logger backend for saving training metrics to disk.

Provides `CSVLogger` class to append step metrics to a structured CSV file.
"""

from __future__ import annotations

import csv
import os
from typing import Dict

from stackformer.utils.utils import is_main_process


class CSVLogger:
    """Writes numeric scalar training metrics to a CSV file on disk.

    Constructor args:
        log_dir (str, default="logs"): Output directory path for log files.
        filename (str, default="metrics.csv"): Target CSV filename.
    """

    def __init__(self, log_dir: str = "logs", filename: str = "metrics.csv") -> None:
        self.enabled = is_main_process()
        self.file = None
        self.writer = None
        self.headers_written = False

        if not self.enabled:
            return

        os.makedirs(log_dir, exist_ok=True)
        self.path = os.path.join(log_dir, filename)
        if os.path.exists(self.path) and os.path.getsize(self.path) > 0:
            self.headers_written = True
        self.file = open(self.path, mode="a", newline="", encoding="utf-8")

    def log(self, metrics: Dict[str, float]) -> None:
        """Write key-value metrics row to CSV file.

        Args:
            metrics (Dict[str, float]): Key-value pair metric dictionary.
        """
        if not self.enabled or not metrics:
            return
        if self.file is None:
            return

        clean = {k: v for k, v in metrics.items() if isinstance(v, (int, float))}
        if not clean:
            return

        if self.writer is None:
            self.writer = csv.DictWriter(self.file, fieldnames=list(clean.keys()), extrasaction="ignore")

        if not self.headers_written:
            self.writer.writeheader()
            self.headers_written = True

        self.writer.writerow(clean)

    def flush(self) -> None:
        """Flush unwritten CSV buffers to disk."""
        if self.enabled and self.file:
            self.file.flush()

    def close(self) -> None:
        """Flush and close active CSV file handle."""
        if self.enabled and self.file:
            self.file.flush()
            self.file.close()
            self.file = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

