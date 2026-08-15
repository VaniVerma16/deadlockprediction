from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


LOCK_ACQUIRED = {"lock_acquired", "trylock_acquired"}
LOCK_ATTEMPTED = {"lock_attempt", "trylock_attempt"}
LOCK_RELEASED = {"lock_released"}
WAIT_STARTED = {"futex_wait"}
WAIT_ENDED = {"futex_return", "futex_wake"}
MUTEX_EVENTS = LOCK_ATTEMPTED | LOCK_ACQUIRED | LOCK_RELEASED
EDGE_EVENTS = LOCK_ACQUIRED | LOCK_RELEASED | WAIT_STARTED | WAIT_ENDED


def read_events(path: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            events.append(event)
    return sorted(events, key=lambda event: (event["ts_ns"], event.get("seq", 0)))


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")


def _lock_key(event: dict[str, Any]) -> str | None:
    address = event.get("lock_addr")
    pid = event.get("pid")
    if address in (None, "0x0", "0x0000000000000000") or pid is None:
        return None
    return f"lock:{pid}:{address.lower()}"


def _thread_key(event: dict[str, Any]) -> str | None:
    pid = event.get("pid")
    tid = event.get("tid")
    if pid is None or tid is None:
        return None
    return f"thread:{pid}:{tid}"


@dataclass
class GraphState:
    owners: dict[str, str] = field(default_factory=dict)
    waiters: dict[str, tuple[str, int]] = field(default_factory=dict)
    threads: set[str] = field(default_factory=set)
    locks: set[str] = field(default_factory=set)
    known_mutexes: set[str] = field(default_factory=set)
    allowed_mutexes: set[str] | None = None
    scheduler_switches: dict[str, int] = field(default_factory=dict)
    wakeups: dict[str, int] = field(default_factory=dict)
    last_cpu: dict[str, int] = field(default_factory=dict)
    cpu_migrations: dict[str, int] = field(default_factory=dict)

    def apply(self, event: dict[str, Any]) -> None:
        event_type = event["event"]
        thread = _thread_key(event)
        lock = _lock_key(event)
        timestamp = event["ts_ns"]

        mutex_allowed = lock is not None and (
            self.allowed_mutexes is None or lock in self.allowed_mutexes
        )

        if event_type in MUTEX_EVENTS and thread and mutex_allowed:
            self.threads.add(thread)
            self.locks.add(lock)
            self.known_mutexes.add(lock)

        if event_type in LOCK_ACQUIRED and thread and mutex_allowed and event.get("ret", 0) == 0:
            self.owners[lock] = thread
            self.waiters.pop(thread, None)
        elif event_type in LOCK_RELEASED and thread and mutex_allowed:
            if self.owners.get(lock) == thread:
                self.owners.pop(lock, None)
        elif event_type in WAIT_STARTED and thread and lock in self.known_mutexes:
            self.threads.add(thread)
            self.waiters[thread] = (lock, timestamp)
        elif event_type in WAIT_ENDED and thread:
            waiting = self.waiters.get(thread)
            if waiting and (lock is None or waiting[0] == lock):
                self.waiters.pop(thread, None)
        elif event_type == "sched_switch" and thread in self.threads:
            self.scheduler_switches[thread] = self.scheduler_switches.get(thread, 0) + 1
            cpu = event.get("cpu")
            if isinstance(cpu, int):
                previous_cpu = self.last_cpu.get(thread)
                if previous_cpu is not None and previous_cpu != cpu:
                    self.cpu_migrations[thread] = self.cpu_migrations.get(thread, 0) + 1
                self.last_cpu[thread] = cpu
        elif event_type == "thread_wakeup":
            target_tid = event.get("target_tid")
            target = next(
                (candidate for candidate in self.threads if candidate.endswith(f":{target_tid}")),
                None,
            )
            if target is not None:
                self.wakeups[target] = self.wakeups.get(target, 0) + 1
        elif event_type == "thread_exit" and thread:
            self.waiters.pop(thread, None)
            self.threads.discard(thread)
            self.scheduler_switches.pop(thread, None)
            self.wakeups.pop(thread, None)
            self.last_cpu.pop(thread, None)
            self.cpu_migrations.pop(thread, None)
            for owned_lock, owner in list(self.owners.items()):
                if owner == thread:
                    self.owners.pop(owned_lock, None)

    def cycle_nodes(self) -> list[str]:
        adjacency: dict[str, list[str]] = {}
        for lock, thread in self.owners.items():
            adjacency.setdefault(lock, []).append(thread)
        for thread, (lock, _) in self.waiters.items():
            adjacency.setdefault(thread, []).append(lock)

        index = 0
        indices: dict[str, int] = {}
        lowlinks: dict[str, int] = {}
        stack: list[str] = []
        on_stack: set[str] = set()
        cycles: list[list[str]] = []

        def strong_connect(node: str) -> None:
            nonlocal index
            indices[node] = index
            lowlinks[node] = index
            index += 1
            stack.append(node)
            on_stack.add(node)

            for neighbor in adjacency.get(node, []):
                if neighbor not in indices:
                    strong_connect(neighbor)
                    lowlinks[node] = min(lowlinks[node], lowlinks[neighbor])
                elif neighbor in on_stack:
                    lowlinks[node] = min(lowlinks[node], indices[neighbor])

            if lowlinks[node] == indices[node]:
                component: list[str] = []
                while True:
                    member = stack.pop()
                    on_stack.remove(member)
                    component.append(member)
                    if member == node:
                        break
                if len(component) > 1:
                    cycles.append(sorted(component))

        for node in sorted(set(adjacency) | {n for values in adjacency.values() for n in values}):
            if node not in indices:
                strong_connect(node)

        return min(cycles) if cycles else []

    def snapshot(self, run_id: str, timestamp: int) -> dict[str, Any]:
        cycle = self.cycle_nodes()
        nodes: list[dict[str, Any]] = []
        for thread in sorted(self.threads):
            waiting = self.waiters.get(thread)
            nodes.append({
                "id": thread,
                "type": "thread",
                "features": {
                    "is_waiting": int(waiting is not None),
                    "wait_ns": max(0, timestamp - waiting[1]) if waiting else 0,
                    "scheduler_switches": self.scheduler_switches.get(thread, 0),
                    "wakeups": self.wakeups.get(thread, 0),
                    "cpu_migrations": self.cpu_migrations.get(thread, 0),
                    "last_cpu": self.last_cpu.get(thread, -1),
                },
            })
        for lock in sorted(self.locks):
            nodes.append({
                "id": lock,
                "type": "lock",
                "features": {"has_owner": int(lock in self.owners)},
            })

        edges = [
            {"source": lock, "target": thread, "type": "owned_by"}
            for lock, thread in sorted(self.owners.items())
        ]
        edges.extend(
            {"source": thread, "target": lock, "type": "waits_for"}
            for thread, (lock, _) in sorted(self.waiters.items())
        )
        return {
            "run_id": run_id,
            "ts_ns": timestamp,
            "nodes": nodes,
            "edges": edges,
            "cycle_nodes": cycle,
            "has_cycle": bool(cycle),
        }


def build_snapshots(
    events: list[dict[str, Any]],
    interval_ns: int,
    unsafe_window_ns: int,
    confirm_ns: int,
    graph_features: dict[str, Any] | None = None,
    start_policy: str = "run",
    event_aligned: bool = False,
    post_cycle_ns: int | None = None,
    drop_empty: bool = False,
    drop_edgeless: bool = False,
    deduplicate: bool = False,
) -> list[dict[str, Any]]:
    if not events:
        return []
    run_ids = {event["run_id"] for event in events}
    if len(run_ids) != 1:
        raise ValueError("one build invocation must contain exactly one run_id")
    run_id = next(iter(run_ids))
    provenance_source = next(
        (event for event in events if event.get("source") == "workload"), events[0]
    )
    provenance = {
        key: provenance_source[key]
        for key in ("scenario", "mode", "seed")
        if key in provenance_source
    }
    ground_truth_mutexes = {
        lock for event in events
        if event.get("source") == "workload"
        and event.get("event", "").startswith("ground_truth_lock_")
        and (lock := _lock_key(event)) is not None
    }
    allowed_mutexes = ground_truth_mutexes or None

    def graph_event_is_relevant(event: dict[str, Any]) -> bool:
        if event["event"] not in MUTEX_EVENTS | WAIT_STARTED | WAIT_ENDED:
            return False
        lock = _lock_key(event)
        return lock is not None and (allowed_mutexes is None or lock in allowed_mutexes)

    if start_policy == "run":
        start = min(event["ts_ns"] for event in events)
    elif start_policy == "first_mutex_event":
        mutex_times = [
            event["ts_ns"] for event in events
            if event["event"] in MUTEX_EVENTS
            and _thread_key(event)
            and graph_event_is_relevant(event)
        ]
        if not mutex_times:
            return []
        start = min(mutex_times)
    elif start_policy == "first_graph_edge":
        edge_times = [
            event["ts_ns"] for event in events
            if event["event"] in LOCK_ACQUIRED | WAIT_STARTED
            and _thread_key(event)
            and graph_event_is_relevant(event)
        ]
        if not edge_times:
            return []
        start = min(edge_times)
    else:
        raise ValueError(f"unknown start policy: {start_policy}")
    explicit_end = [event["ts_ns"] for event in events if event["event"] == "run_end"]
    end = max(explicit_end or [event["ts_ns"] for event in events])

    timestamps = set(range(start, end + 1, interval_ns))
    timestamps.add(end)
    if event_aligned:
        timestamps.update(
            event["ts_ns"] for event in events
            if start <= event["ts_ns"] <= end
            and event["event"] in EDGE_EVENTS
            and graph_event_is_relevant(event)
        )

    state = GraphState(allowed_mutexes=allowed_mutexes)
    snapshots: list[dict[str, Any]] = []
    event_index = 0
    for timestamp in sorted(timestamps):
        while event_index < len(events) and events[event_index]["ts_ns"] <= timestamp:
            state.apply(events[event_index])
            event_index += 1
        snapshot = state.snapshot(run_id, timestamp)
        if drop_empty and not snapshot["nodes"] and not snapshot["edges"]:
            continue
        if drop_edgeless and not snapshot["edges"]:
            continue
        if not drop_empty or snapshot["nodes"] or snapshot["edges"]:
            snapshots.append(snapshot)

    cycle_times = [snapshot["ts_ns"] for snapshot in snapshots if snapshot["has_cycle"]]
    first_cycle = min(cycle_times) if cycle_times else None
    if first_cycle is not None and post_cycle_ns is not None:
        snapshots = [
            snapshot for snapshot in snapshots
            if snapshot["ts_ns"] <= first_cycle + post_cycle_ns
        ]
    for snapshot in snapshots:
        ts_ns = snapshot["ts_ns"]
        if snapshot["has_cycle"] and first_cycle is not None and ts_ns >= first_cycle + confirm_ns:
            label = "deadlocked"
        elif first_cycle is not None and first_cycle - unsafe_window_ns <= ts_ns < first_cycle + confirm_ns:
            label = "unsafe"
        else:
            label = "safe"
        snapshot["label"] = label
        snapshot["label_metadata"] = {
            "first_cycle_ts_ns": first_cycle,
            "unsafe_window_ns": unsafe_window_ns,
            "confirmation_ns": confirm_ns,
            "start_policy": start_policy,
            "event_aligned": event_aligned,
            "post_cycle_ns": post_cycle_ns,
            "mutex_filter": "ground_truth" if allowed_mutexes is not None else "observed",
            "drop_empty": drop_empty,
            "drop_edgeless": drop_edgeless,
            "deduplicated": deduplicate,
        }
        snapshot["graph_features"] = graph_features or {}
        snapshot["provenance"] = provenance
    if deduplicate:
        unique_snapshots: list[dict[str, Any]] = []
        previous_signature: str | None = None
        for snapshot in snapshots:
            signature = json.dumps({
                "nodes": snapshot["nodes"],
                "edges": snapshot["edges"],
                "label": snapshot["label"],
            }, sort_keys=True, separators=(",", ":"))
            if signature != previous_signature:
                unique_snapshots.append(snapshot)
                previous_signature = signature
        snapshots = unique_snapshots
    return snapshots
