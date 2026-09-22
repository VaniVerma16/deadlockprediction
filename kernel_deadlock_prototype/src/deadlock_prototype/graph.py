from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import networkx as nx
import numpy as np

from .events import RuntimeEvent
from .schema import Snapshot, validate_snapshot


def _event_dict(event: RuntimeEvent | dict) -> dict:
    return event.to_dict() if isinstance(event, RuntimeEvent) else event


def build_snapshot(
    events: Iterable[RuntimeEvent | dict],
    snapshot_id: str = "snapshot-0000",
    snapshot_ts_ns: int | None = None,
) -> Snapshot:
    """Reconstruct one synchronization graph from normalized events.

    If snapshot_ts_ns is provided, it represents the temporal sampling point
    of the Digital Twin. Waiting times are measured relative to that point.
    """

    ordered = sorted(
        (_event_dict(event) for event in events),
        key=lambda item: item["ts_ns"],
    )

    if not ordered:
        raise ValueError("cannot build a snapshot from no events")

    owners: dict[str, int] = {}
    waits: dict[int, dict] = {}

    thread_stats: defaultdict[int, dict[str, float]] = defaultdict(
        lambda: {
            "scheduler_switches": 0,
            "wakeups": 0,
            "cpu_migrations": 0,
            "last_cpu": -1,
        }
    )

    locks: set[str] = set()
    threads: set[int] = set()

    # The snapshot timestamp is supplied by the Digital Twin when doing
    # fixed-interval temporal sampling. Otherwise use the latest event time.
    latest_ts = (
        int(snapshot_ts_ns)
        if snapshot_ts_ns is not None
        else int(ordered[-1]["ts_ns"])
    )

    for event in ordered:
        # Only process events that have occurred by this snapshot time.
        if int(event["ts_ns"]) > latest_ts:
            continue

        tid = int(event["tid"])
        event_name = event["event"]
        lock_id = event.get("lock_id")

        threads.add(tid)

        if lock_id:
            locks.add(lock_id)

        if event_name == "lock_wait_start" and lock_id:
            waits[tid] = {
                "lock_id": lock_id,
                "start_ns": int(event["ts_ns"]),
            }

        elif event_name == "lock_acquired" and lock_id:
            owners[lock_id] = tid
            waits.pop(tid, None)

            thread_stats[tid]["wakeups"] += float(
                event.get("wakeups", 0)
            )

        elif (
            event_name == "lock_released"
            and lock_id
            and owners.get(lock_id) == tid
        ):
            owners.pop(lock_id, None)

        elif event_name in {"lock_wait_timeout", "lock_wait_end"}:
            waits.pop(tid, None)

        elif event_name == "sched_switch":
            thread_stats[tid]["scheduler_switches"] += 1

        elif event_name == "thread_wakeup":
            thread_stats[tid]["wakeups"] += 1

    graph = nx.DiGraph()

    # Thread nodes
    for tid in sorted(threads):
        waiting = waits.get(tid)

        if waiting:
            wait_ns = max(
                0,
                latest_ts - int(waiting["start_ns"]),
            )
        else:
            wait_ns = 0

        thread_stats[tid]["wait_ns"] = wait_ns
        thread_stats[tid]["is_waiting"] = int(waiting is not None)

        graph.add_node(
            f"thread:{tid}",
            type="thread",
            features=dict(thread_stats[tid]),
        )

    # Lock nodes
    for lock_id in sorted(locks):
        graph.add_node(
            lock_id,
            type="lock",
            features={
                "has_owner": int(lock_id in owners),
            },
        )

    # Ownership edges: lock -> thread
    for lock_id, tid in owners.items():
        graph.add_edge(
            lock_id,
            f"thread:{tid}",
            type="owned_by",
        )

    # Waiting edges: thread -> lock
    for tid, waiting in waits.items():
        graph.add_edge(
            f"thread:{tid}",
            waiting["lock_id"],
            type="waits_for",
        )

    cycles = list(nx.simple_cycles(graph))

    max_wait_ns = max(
    (
        data["features"].get("wait_ns", 0)
        for _, data in graph.nodes(data=True)
        if data["type"] == "thread"
    ),
    default=0,
    )

    metadata = {
        "cycle_count": len(cycles),
        "max_wait_ns": int(max_wait_ns),
        "feature_vector": np.asarray(
            [
                len(threads),
                len(locks),
                graph.number_of_edges(),
                max_wait_ns,
            ],
            dtype=float,
        ).tolist(),
        "source": "python_tracked_lock_demo",
    }

    if cycles:
        metadata["rule_label"] = "deadlocked"

    elif (
        max_wait_ns >= 50_000_000
        or any(
            data["type"] == "thread"
            and data.get("is_waiting")
            for _, data in graph.nodes(data=True)
        )
    ):
        metadata["rule_label"] = "pre_deadlock"

    else:
        metadata["rule_label"] = "safe"

    raw = {
        "snapshot_id": snapshot_id,
        "ts_ns": latest_ts,
        "nodes": [
            {
                "id": node_id,
                "type": data["type"],
                "features": data["features"],
            }
            for node_id, data in graph.nodes(data=True)
        ],
        "edges": [
            {
                "source": source,
                "target": target,
                "type": data["type"],
            }
            for source, target, data in graph.edges(data=True)
        ],
        "metadata": metadata,
    }

    return validate_snapshot(raw)

def build_snapshot_sequence(
    events: Iterable[RuntimeEvent | dict],
    snapshot_interval_ns: int = 10_000_000,
    min_snapshots: int = 8,
) -> list[Snapshot]:
    """Build snapshots at fixed temporal intervals.

    The Digital Twin uses fixed 10-ms sampling so that temporal sequences
    are compatible with the temporal model training data.
    """

    ordered = sorted(
        (_event_dict(event) for event in events),
        key=lambda item: item["ts_ns"],
    )

    if not ordered:
        return []

    if snapshot_interval_ns <= 0:
        raise ValueError("snapshot_interval_ns must be positive")

    if min_snapshots < 1:
        raise ValueError("min_snapshots must be at least 1")

    start_ts = int(ordered[0]["ts_ns"])
    end_ts = int(ordered[-1]["ts_ns"])

    # Generate fixed 10-ms sampling points.
    snapshot_times = []

    ts = start_ts
    while ts <= end_ts:
        snapshot_times.append(ts)
        ts += snapshot_interval_ns

    # Make sure short test runs still produce at least min_snapshots.
    while len(snapshot_times) < min_snapshots:
        snapshot_times.append(
            start_ts + len(snapshot_times) * snapshot_interval_ns
        )

    snapshots = []

    for index, snapshot_ts in enumerate(snapshot_times):
        prefix = [
            event
            for event in ordered
            if int(event["ts_ns"]) <= snapshot_ts
        ]

        # If no event has occurred yet at this sampling point,
        # use all events seen so far only when available.
        if not prefix:
            continue

        snapshots.append(
            build_snapshot(
                prefix,
                snapshot_id=f"snapshot-{index:04d}",
                snapshot_ts_ns=snapshot_ts,
            )
        )

    return snapshots

def write_snapshot_json(snapshot: Snapshot, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = snapshot.model_dump() if hasattr(snapshot, "model_dump") else snapshot.dict()
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
