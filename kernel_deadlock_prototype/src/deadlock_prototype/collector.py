from __future__ import annotations

from pathlib import Path

from .events import EventLogger
from .workload import WORKLOADS


def collect_workload_events(mode: str, output_path: Path) -> int:
    """Run the VM workload and persist only its raw normalized event stream."""

    try:
        workload = WORKLOADS[mode]
    except KeyError as error:
        raise ValueError(f"unknown workload mode: {mode}") from error

    logger = EventLogger()
    workload(logger)
    logger.write_jsonl(output_path)
    return len(logger.events)