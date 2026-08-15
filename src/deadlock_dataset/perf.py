from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


def read_perf_stat(path: Path) -> dict[str, Any]:
    counters: dict[str, Any] = {}
    if not path.exists():
        return counters
    with path.open(encoding="utf-8") as handle:
        for row in csv.reader(handle):
            if len(row) < 3 or not row[2] or row[0].startswith("#"):
                continue
            raw_value = row[0].strip()
            event_name = row[2].strip().replace("-", "_")
            if raw_value in {"<not counted>", "<not supported>"}:
                counters[event_name] = None
                continue
            try:
                counters[event_name] = float(raw_value.replace(" ", ""))
            except ValueError:
                continue
    return counters

