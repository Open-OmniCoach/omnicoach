"""`RunTelemetry`, which periodically writes provider and media counters for a run."""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any

from ..artifacts.store import write_json_atomic
from ..providers.telemetry import snapshot


class RunTelemetry:
    def __init__(self, root: str | Path, media: Any) -> None:
        self.root = Path(root)
        self.media = media
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

    def write(self) -> None:
        write_json_atomic(
            self.root / "runtime_metrics.json",
            {
                "timestamp": time.time(),
                "provider": snapshot(),
                "media": self.media.snapshot() if self.media else {},
            },
        )

    def run(self) -> None:
        while not self.stop.wait(10):
            self.write()

    def close(self) -> None:
        self.stop.set()
        self.thread.join()
        self.write()
