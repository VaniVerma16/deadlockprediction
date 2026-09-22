from __future__ import annotations

import json
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, TextIO


@dataclass(slots=True)
class RuntimeEvent:
    """One normalized event consumed by the graph builder."""

    ts_ns: int
    event: str
    tid: int
    lock_id: str | None = None
    owner_tid: int | None = None
    wait_ns: int = 0
    wakeups: int = 0
    context_switches: int = 0
    cpu_migrations: int = 0
    cpu: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class EventLogger:
    """Thread-safe JSONL event logger used by the toy workload."""

    def __init__(self, output: TextIO | None = None) -> None:
        self.output = output
        self.events: list[RuntimeEvent] = []
        self._mutex = threading.Lock()

    def emit(self, event: RuntimeEvent) -> None:
        with self._mutex:
            self.events.append(event)
            if self.output is not None:
                self.output.write(json.dumps(event.to_dict()) + "\n")
                self.output.flush()

    def record(self, event: str, tid: int, **kwargs: Any) -> None:
        self.emit(RuntimeEvent(ts_ns=time.monotonic_ns(), event=event, tid=tid, **kwargs))

    def write_jsonl(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            for event in self.events:
                handle.write(json.dumps(event.to_dict()) + "\n")


class TrackedLock:
    """A lock wrapper that records ownership, waits, releases, and wakeups."""

    def __init__(self, lock_id: str, logger: EventLogger) -> None:
        self.lock_id = lock_id
        self.logger = logger
        self._lock = threading.Lock()
        self._state_lock = threading.Lock()
        self.owner_tid: int | None = None
        self._waiting = 0

    def acquire(self, timeout: float | None = None) -> bool:
        tid = threading.get_native_id()
        start = time.monotonic_ns()
        with self._state_lock:
            self._waiting += 1
            waiting = self._waiting
        self.logger.record("lock_wait_start", tid, lock_id=self.lock_id, owner_tid=self.owner_tid)
        acquired = self._lock.acquire() if timeout is None else self._lock.acquire(timeout=timeout)
        wait_ns = time.monotonic_ns() - start
        with self._state_lock:
            self._waiting -= 1
        if acquired:
            with self._state_lock:
                self.owner_tid = tid
            self.logger.record(
                "lock_acquired",
                tid,
                lock_id=self.lock_id,
                owner_tid=tid,
                wait_ns=wait_ns,
                wakeups=max(0, waiting - 1),
            )
        else:
            self.logger.record(
                "lock_wait_timeout",
                tid,
                lock_id=self.lock_id,
                owner_tid=self.owner_tid,
                wait_ns=wait_ns,
            )
        return acquired

    def release(self) -> None:
        tid = threading.get_native_id()
        with self._state_lock:
            if self.owner_tid != tid:
                raise RuntimeError(f"{self.lock_id} is not owned by thread {tid}")
            self.owner_tid = None
        self._lock.release()
        self.logger.record("lock_released", tid, lock_id=self.lock_id)

    def __enter__(self) -> "TrackedLock":
        if not self.acquire():
            raise RuntimeError(f"could not acquire {self.lock_id}")
        return self

    def __exit__(self, *_: object) -> None:
        self.release()
