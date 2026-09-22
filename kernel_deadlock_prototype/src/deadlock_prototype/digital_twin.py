from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Iterable

from .events import RuntimeEvent
from .graph import build_snapshot


@dataclass
class DigitalTwinConfig:
    """Runtime configuration for the synchronization Digital Twin."""

    snapshot_interval_ns: int = 10_000_000  # 10 ms


class SynchronizationDigitalTwin:
    """
    Continuously updated software representation of synchronization state.

    Runtime events are retained as the source of truth. At each 10-ms
    sampling boundary, the event history is replayed into the existing graph
    reconstruction logic to produce a Temporal Synchronization Graph snapshot.

    This is intentionally a prototype implementation. It favors correctness
    and transparency over a fully incremental state engine.
    """

    def __init__(self, config: DigitalTwinConfig | None = None):
        self.config = config or DigitalTwinConfig()
        self.events: deque[RuntimeEvent] = deque()
        self.start_ts_ns: int | None = None
        self.next_snapshot_ts_ns: int | None = None

    def update(self, event: RuntimeEvent) -> None:
        """Add one normalized runtime event to the Digital Twin."""
        if self.start_ts_ns is None:
            self.start_ts_ns = event.ts_ns
            self.next_snapshot_ts_ns = event.ts_ns

        self.events.append(event)

    def update_many(self, events: Iterable[RuntimeEvent]) -> None:
        for event in events:
            self.update(event)

    def ready(self, now_ns: int) -> bool:
        """Return True when a new fixed-rate snapshot is due."""
        return (
            self.next_snapshot_ts_ns is not None
            and now_ns >= self.next_snapshot_ts_ns
        )

    def sample(self, snapshot_id: str):
        """
        Create the next 10-ms TSG snapshot.

        Multiple events may have occurred since the previous sample; all
        events up to the sampling boundary are represented in the state.
        """
        if self.next_snapshot_ts_ns is None:
            return None

        sample_ts = self.next_snapshot_ts_ns
        events = list(self.events)

        if not events:
            return None

        snapshot = build_snapshot(
            events,
            snapshot_id=snapshot_id,
            snapshot_ts_ns=sample_ts,
        )

        self.next_snapshot_ts_ns += self.config.snapshot_interval_ns
        return snapshot

    def sample_until(self, now_ns: int):
        """
        Generate all snapshots whose sampling boundaries have elapsed.
        """
        snapshots = []
        counter = 0

        while self.ready(now_ns):
            snapshot = self.sample(f"live-{counter:06d}")
            if snapshot is None:
                break
            snapshots.append(snapshot)
            counter += 1

        return snapshots

    @property
    def current_time_ns(self) -> int | None:
        return self.next_snapshot_ts_ns
