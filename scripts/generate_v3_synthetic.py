#!/usr/bin/env python3
"""Generate Temporal Synchronization Graph Dataset v3.

The generator creates hard synthetic thread-mutex graph traces for temporal
deadlock prediction.  It intentionally overlaps wait durations and graph sizes
across classes, includes transient/recovered cycles, and computes labels from
future persistent-cycle outcomes only after each full run has been generated.

The emitted JSONL files retain the v2 node and edge input contract.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


SCENARIOS = (
    "two_process_deadlock",
    "cycle_3_5",
    "long_chain_no_cycle",
    "almost_cycle_pre_deadlock",
    "multiple_cycles",
    "cycle_with_safe_processes",
    "resource_contention_no_deadlock",
    "same_graph_different_states",
    "delayed_deadlock",
    "deadlock_recovery",
    "imbalanced_resource_allocation",
)

LABELS = ("safe", "pre_deadlock", "deadlocked")
INTERVAL_NS = 10_000_000
SEQUENCE_LENGTH = 8
CONFIRM_STEPS = 5
PREDICTION_HORIZON_STEPS = 30


@dataclass
class GraphState:
    owners: dict[int, int]
    waits: dict[int, int]


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")


def add_chain(
    owners: dict[int, int],
    waits: dict[int, int],
    size: int,
    wait_edges: int,
    close_cycle: bool,
    thread_offset: int = 0,
    lock_offset: int = 0,
) -> None:
    """Add lock_i -> thread_i -> lock_(i+1) dependencies."""
    for index in range(size):
        owners[lock_offset + index] = thread_offset + index
    for index in range(min(wait_edges, size - 1)):
        waits[thread_offset + index] = lock_offset + index + 1
    if close_cycle:
        waits[thread_offset + size - 1] = lock_offset


def add_background_contention(
    owners: dict[int, int],
    waits: dict[int, int],
    num_threads: int,
    num_locks: int,
    t: int,
    rng: random.Random,
    reserved_threads: int,
    reserved_locks: int,
) -> None:
    """Add unrelated acyclic activity without changing the primary pattern."""
    available_threads = list(range(reserved_threads, num_threads))
    available_locks = list(range(reserved_locks, num_locks))
    if not available_threads or not available_locks:
        return

    owner = available_threads[(t // 12) % len(available_threads)]
    lock = available_locks[(t // 17) % len(available_locks)]
    owners.setdefault(lock, owner)
    if len(available_threads) > 1 and (t // 6) % 3 != 0:
        waiter = available_threads[(available_threads.index(owner) + 1) % len(available_threads)]
        if waiter != owner and waiter not in waits:
            waits[waiter] = lock

    if rng.random() < 0.18 and len(available_locks) > 1:
        second_lock = available_locks[(available_locks.index(lock) + 1) % len(available_locks)]
        second_owner = available_threads[-1]
        owners.setdefault(second_lock, second_owner)


def choose_dimensions(scenario: str, split: str, rng: random.Random) -> tuple[int, int, int]:
    """Return thread count, lock count, and primary cycle/chain size."""
    if scenario == "two_process_deadlock":
        primary = 2
    elif scenario in {"cycle_3_5", "multiple_cycles"}:
        primary = rng.randint(3, 5)
    elif scenario in {"long_chain_no_cycle", "delayed_deadlock"}:
        primary = rng.randint(4, 7 if split == "train" else 9)
    else:
        primary = rng.randint(2, 5)

    extra_ranges = {
        "train": (0, 4),
        "validation": (2, 6),
        "test": (4, 8),
    }
    extra = rng.randint(*extra_ranges[split])
    threads = max(primary + extra, 5 if scenario == "resource_contention_no_deadlock" else primary)
    threads = min(16, threads)
    locks = min(14, max(primary + rng.randint(1, 5), 3))

    if scenario == "multiple_cycles":
        threads = max(threads, primary + 3)
        locks = max(locks, primary + 3)
    if scenario == "imbalanced_resource_allocation":
        locks = max(8, min(14, threads + rng.randint(2, 5)))
    return threads, locks, primary


def build_states(
    scenario: str,
    length: int,
    num_threads: int,
    num_locks: int,
    primary: int,
    rng: random.Random,
) -> tuple[list[GraphState], dict]:
    """Create one temporal graph trace without assigning labels."""
    states: list[GraphState] = []
    outcome = rng.random()
    build_start = rng.randint(18, 36)
    cycle_at = rng.randint(max(build_start + 24, 55), min(length - 25, 100))
    recovery_at = min(length - 8, cycle_at + rng.randint(20, 42))
    variant = "persistent" if outcome < 0.68 else "resolved"

    for t in range(length):
        owners: dict[int, int] = {}
        waits: dict[int, int] = {}
        reserved_threads = primary
        reserved_locks = primary

        if scenario == "two_process_deadlock":
            owners[0] = 0
            owners[1] = 1
            if t >= build_start:
                waits[0] = 1
            if t >= cycle_at:
                waits[1] = 0
            if variant == "resolved" and t >= cycle_at + rng.randint(2, 4):
                waits.pop(1, None)

        elif scenario == "cycle_3_5":
            progress = max(0, min(primary - 1, (t - build_start) // 7 + 1)) if t >= build_start else 0
            close = t >= cycle_at
            add_chain(owners, waits, primary, progress, close)
            if variant == "resolved" and t >= cycle_at + 3:
                waits.pop(primary - 1, None)

        elif scenario == "long_chain_no_cycle":
            progress = max(1, min(primary - 1, (t - 8) // 8 + 1))
            add_chain(owners, waits, primary, progress, False)
            if 40 <= t < 75 and primary < num_threads and primary < num_locks:
                owners[primary] = primary
                waits[primary] = primary - 1

        elif scenario == "almost_cycle_pre_deadlock":
            add_chain(owners, waits, primary, primary - 1, False)
            if variant == "persistent" and t >= cycle_at:
                waits[primary - 1] = 0
            elif variant == "resolved" and t >= cycle_at:
                waits.pop(max(0, primary - 2), None)

        elif scenario == "multiple_cycles":
            first_size = 2
            second_size = 3
            add_chain(owners, waits, first_size, first_size - 1, 24 <= t < 28)
            add_chain(
                owners,
                waits,
                second_size,
                second_size - 1,
                t >= cycle_at,
                thread_offset=primary,
                lock_offset=primary,
            )
            if t >= cycle_at + 12:
                waits[1] = 0
            reserved_threads = primary + second_size
            reserved_locks = primary + second_size

        elif scenario == "cycle_with_safe_processes":
            cycle_size = min(primary, 3)
            add_chain(owners, waits, cycle_size, cycle_size - 1, t >= cycle_at)
            reserved_threads = cycle_size
            reserved_locks = cycle_size
            if variant == "resolved" and t >= cycle_at + 4:
                waits.pop(cycle_size - 1, None)

        elif scenario == "resource_contention_no_deadlock":
            owner = (t // 28) % num_threads
            owners[0] = owner
            waiting_count = min(num_threads - 1, 2 + (t // 12) % max(2, num_threads - 1))
            for offset in range(1, waiting_count + 1):
                waiter = (owner + offset) % num_threads
                waits[waiter] = 0
            for lock in range(1, min(num_locks, 4)):
                owners[lock] = (owner + lock + 1) % num_threads
            reserved_threads = num_threads
            reserved_locks = min(num_locks, 4)

        elif scenario == "same_graph_different_states":
            reference = cycle_at - 8
            add_chain(owners, waits, primary, primary - 1, False)
            if variant == "persistent":
                # The chain slowly forms and closes after the shared reference state.
                if t < reference:
                    keep = max(1, min(primary - 1, (t - 10) // 9 + 1))
                    waits.clear()
                    for index in range(keep):
                        waits[index] = index + 1
                if t >= cycle_at:
                    waits[primary - 1] = 0
            else:
                # It reaches the same almost-cycle after a short transient cycle,
                # then backs off instead of becoming persistently deadlocked.
                if reference - 18 <= t < reference - 14:
                    waits[primary - 1] = 0
                if t >= cycle_at:
                    waits.pop(primary - 2, None)

        elif scenario == "delayed_deadlock":
            progress = max(0, min(primary - 1, (t - build_start) // 10 + 1)) if t >= build_start else 0
            add_chain(owners, waits, primary, progress, t >= cycle_at)
            if variant == "resolved" and t >= cycle_at + 3:
                waits.pop(primary - 1, None)

        elif scenario == "deadlock_recovery":
            add_chain(owners, waits, primary, primary - 1, cycle_at <= t < recovery_at)
            if t >= recovery_at:
                waits.pop(primary - 1, None)
                if t >= recovery_at + 10:
                    waits.pop(0, None)
            elif t < cycle_at:
                waits.pop(primary - 1, None)
            variant = "confirmed_then_recovered"

        elif scenario == "imbalanced_resource_allocation":
            dominant = 0
            for lock in range(num_locks):
                owners[lock] = dominant if lock < num_locks - 2 else min(lock % num_threads, num_threads - 1)
            for thread in range(1, min(num_threads, 7)):
                waits[thread] = thread % max(1, num_locks - 2)
            if variant == "persistent" and t >= cycle_at:
                waits[dominant] = num_locks - 1
                owners[num_locks - 1] = 1
            reserved_threads = min(num_threads, 7)
            reserved_locks = num_locks

        else:
            raise ValueError(f"Unknown scenario: {scenario}")

        add_background_contention(
            owners,
            waits,
            num_threads,
            num_locks,
            t,
            rng,
            min(reserved_threads, num_threads),
            min(reserved_locks, num_locks),
        )
        states.append(GraphState(owners=owners, waits=waits))

    return states, {
        "outcome_variant": variant,
        "build_start": build_start,
        "planned_cycle_at": cycle_at,
        "planned_recovery_at": recovery_at,
    }


def cycle_components(state: GraphState) -> list[list[tuple[str, int]]]:
    adjacency: dict[tuple[str, int], list[tuple[str, int]]] = defaultdict(list)
    for lock, thread in state.owners.items():
        adjacency[("lock", lock)].append(("thread", thread))
    for thread, lock in state.waits.items():
        adjacency[("thread", thread)].append(("lock", lock))

    index = 0
    stack: list[tuple[str, int]] = []
    indices: dict[tuple[str, int], int] = {}
    low: dict[tuple[str, int], int] = {}
    active: set[tuple[str, int]] = set()
    components: list[list[tuple[str, int]]] = []

    def visit(node: tuple[str, int]) -> None:
        nonlocal index
        indices[node] = index
        low[node] = index
        index += 1
        stack.append(node)
        active.add(node)
        for neighbor in adjacency.get(node, []):
            if neighbor not in indices:
                visit(neighbor)
                low[node] = min(low[node], low[neighbor])
            elif neighbor in active:
                low[node] = min(low[node], indices[neighbor])
        if low[node] == indices[node]:
            component = []
            while True:
                member = stack.pop()
                active.remove(member)
                component.append(member)
                if member == node:
                    break
            if len(component) > 1:
                components.append(sorted(component))

    nodes = set(adjacency)
    nodes.update(neighbor for neighbors in adjacency.values() for neighbor in neighbors)
    for node in sorted(nodes):
        if node not in indices:
            visit(node)
    return sorted(components)


def assign_labels(states: list[GraphState]) -> tuple[list[str], list[bool], list[int], list[bool]]:
    components = [cycle_components(state) for state in states]
    has_cycle = [bool(value) for value in components]
    cycle_count = [len(value) for value in components]
    confirmed = []
    consecutive = 0
    for current in has_cycle:
        consecutive = consecutive + 1 if current else 0
        confirmed.append(consecutive >= CONFIRM_STEPS)

    labels: list[str] = []
    for index in range(len(states)):
        if confirmed[index]:
            labels.append("deadlocked")
        elif any(confirmed[index + 1:index + 1 + PREDICTION_HORIZON_STEPS]):
            labels.append("pre_deadlock")
        else:
            labels.append("safe")
    return labels, has_cycle, cycle_count, confirmed


def snapshot_records(
    run_id: str,
    scenario: str,
    split: str,
    seed: int,
    num_threads: int,
    num_locks: int,
    states: list[GraphState],
    labels: list[str],
    has_cycle: list[bool],
    cycle_count: list[int],
    confirmed: list[bool],
    rng: random.Random,
) -> list[dict]:
    pid = 10_000 + seed % 80_000
    thread_ids = [f"thread:{pid}:{pid + 1 + index}" for index in range(num_threads)]
    lock_ids = [f"lock:{pid}:0x{(seed * 4099 + index * 64):x}" for index in range(num_locks)]
    previous_waits: dict[int, int] = {}
    wait_age: dict[int, int] = {}
    cpus = [rng.randrange(8) for _ in range(num_threads)]
    rows = []

    for t, state in enumerate(states):
        new_wait_age: dict[int, int] = {}
        for thread, lock in state.waits.items():
            if previous_waits.get(thread) == lock:
                age = wait_age.get(thread, 0) + INTERVAL_NS
            else:
                # A trace may begin during an existing wait. Later new waits can
                # only be as old as the interval in which they first appear.
                age = (
                    rng.randint(0, 70) * INTERVAL_NS
                    if t == 0
                    else rng.randint(0, INTERVAL_NS)
                )
            new_wait_age[thread] = age

        nodes = []
        for thread in range(num_threads):
            waiting = thread in state.waits
            if rng.random() < 0.08:
                cpus[thread] = rng.randrange(8)
            nodes.append({
                "id": thread_ids[thread],
                "type": "thread",
                "features": {
                    "is_waiting": int(waiting),
                    "wait_ns": new_wait_age.get(thread, 0),
                    # These are per-interval deltas, not run-age counters.
                    "scheduler_switches": rng.randint(0, 2 if waiting else 6),
                    "wakeups": rng.randint(0, 1 if waiting else 3),
                    "cpu_migrations": int(rng.random() < (0.04 if waiting else 0.12)),
                    "last_cpu": cpus[thread],
                },
            })
        for lock in range(num_locks):
            nodes.append({
                "id": lock_ids[lock],
                "type": "lock",
                "features": {"has_owner": int(lock in state.owners)},
            })

        edges = [
            {"source": lock_ids[lock], "target": thread_ids[thread], "type": "owned_by"}
            for lock, thread in sorted(state.owners.items())
        ]
        edges.extend(
            {"source": thread_ids[thread], "target": lock_ids[lock], "type": "waits_for"}
            for thread, lock in sorted(state.waits.items())
        )
        cycle_nodes = sorted({
            (thread_ids[value] if kind == "thread" else lock_ids[value])
            for component in cycle_components(state)
            for kind, value in component
        })

        next_confirm = next((future for future in range(t + 1, min(len(states), t + 31)) if confirmed[future]), None)
        snapshot_id = f"{run_id}:{t:04d}"
        rows.append({
            "run_id": run_id,
            "snapshot_id": snapshot_id,
            "snapshot_index": t,
            "ts_ns": t * INTERVAL_NS,
            "delta_ns": 0 if t == 0 else INTERVAL_NS,
            "nodes": nodes,
            "edges": edges,
            "cycle_nodes": cycle_nodes,
            "has_cycle": has_cycle[t],
            "label": labels[t],
            "label_metadata": {
                "definition": "future persistent-cycle outcome",
                "confirmation_ms": CONFIRM_STEPS * 10,
                "prediction_horizon_ms": PREDICTION_HORIZON_STEPS * 10,
                "cycle_count": cycle_count[t],
                "confirmed_cycle": confirmed[t],
                "steps_to_next_confirmation": None if next_confirm is None else next_confirm - t,
                "class_vocabulary": list(LABELS),
            },
            "graph_features": {},
            "provenance": {
                "scenario": scenario,
                "split": split,
                "seed": seed,
                "generator": "temporal-synchronization-v3",
            },
        })
        previous_waits = dict(state.waits)
        wait_age = new_wait_age
    return rows


def sequence_records(run: dict, split: str) -> list[dict]:
    snapshots = run["snapshots"]
    stride = 4 if split == "train" else 8
    sequences = []
    for end in range(SEQUENCE_LENGTH - 1, len(snapshots), stride):
        start = end - SEQUENCE_LENGTH + 1
        window = snapshots[start:end + 1]
        end_meta = window[-1]["label_metadata"]
        sequences.append({
            "sequence_id": f"{run['run_id']}:{start:04d}-{end:04d}",
            "run_id": run["run_id"],
            "snapshot_ids": [row["snapshot_id"] for row in window],
            "start_ts_ns": window[0]["ts_ns"],
            "end_ts_ns": window[-1]["ts_ns"],
            "duration_ns": window[-1]["ts_ns"] - window[0]["ts_ns"],
            "label": window[-1]["label"],
            "has_cycle": window[-1]["has_cycle"],
            "targets": {
                "current_confirmed_deadlock": bool(end_meta["confirmed_cycle"]),
                "deadlock_within_50ms": (
                    end_meta["steps_to_next_confirmation"] is not None
                    and end_meta["steps_to_next_confirmation"] <= 5
                ),
                "deadlock_within_100ms": (
                    end_meta["steps_to_next_confirmation"] is not None
                    and end_meta["steps_to_next_confirmation"] <= 10
                ),
                "deadlock_within_300ms": (
                    end_meta["steps_to_next_confirmation"] is not None
                    and end_meta["steps_to_next_confirmation"] <= 30
                ),
            },
            "provenance": {
                "scenario": run["scenario"],
                "split": split,
                "seed": run["seed"],
                "outcome_variant": run["run_metadata"]["outcome_variant"],
                "threads": run["threads"],
                "locks": run["locks"],
            },
        })
    return sequences


def topology_signature(snapshot: dict) -> str:
    node_index = {node["id"]: index for index, node in enumerate(snapshot["nodes"])}
    node_types = tuple(node["type"] for node in snapshot["nodes"])
    canonical = sorted(
        (node_index[edge["source"]], edge["type"], node_index[edge["target"]])
        for edge in snapshot["edges"]
    )
    payload = (node_types, tuple(canonical))
    return hashlib.sha256(repr(payload).encode()).hexdigest()[:16]


def split_for_index(index: int, runs_per_scenario: int) -> str:
    train_end = int(runs_per_scenario * 0.70)
    validation_end = train_end + int(runs_per_scenario * 0.15)
    if index < train_end:
        return "train"
    if index < validation_end:
        return "validation"
    return "test"


def generate(output: Path, runs_per_scenario: int, master_seed: int) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    split_snapshots: dict[str, list[dict]] = defaultdict(list)
    split_sequences: dict[str, list[dict]] = defaultdict(list)
    manifest = []

    for scenario_index, scenario in enumerate(SCENARIOS):
        for local_index in range(runs_per_scenario):
            split = split_for_index(local_index, runs_per_scenario)
            seed = master_seed + scenario_index * 100_000 + local_index * 97
            rng = random.Random(seed)
            num_threads, num_locks, primary = choose_dimensions(scenario, split, rng)
            length = rng.randint(96, 136)
            states, run_metadata = build_states(
                scenario, length, num_threads, num_locks, primary, rng
            )
            labels, has_cycle, cycle_count, confirmed = assign_labels(states)
            run_id = f"v3-{scenario_index:02d}-{local_index:04d}-{seed}"
            snapshots = snapshot_records(
                run_id,
                scenario,
                split,
                seed,
                num_threads,
                num_locks,
                states,
                labels,
                has_cycle,
                cycle_count,
                confirmed,
                rng,
            )
            run = {
                "run_id": run_id,
                "scenario": scenario,
                "seed": seed,
                "threads": num_threads,
                "locks": num_locks,
                "snapshots": snapshots,
                "run_metadata": run_metadata,
            }
            sequences = sequence_records(run, split)
            split_snapshots[split].extend(snapshots)
            split_sequences[split].extend(sequences)
            manifest.append({
                "run_id": run_id,
                "split": split,
                "scenario": scenario,
                "seed": seed,
                "threads": num_threads,
                "locks": num_locks,
                "snapshot_count": len(snapshots),
                "sequence_count": len(sequences),
                **run_metadata,
            })

    for split in ("train", "validation", "test"):
        write_jsonl(output / f"{split}.jsonl", split_snapshots[split])
        write_jsonl(output / f"{split}_sequences.jsonl", split_sequences[split])
    write_jsonl(output / "run_manifest.jsonl", manifest)

    metadata = {
        "dataset_version": "v3",
        "generator_revision": "v3.1-wait-age-consistency",
        "origin": "fully synthetic Python generator; not QEMU/eBPF telemetry",
        "generator_seed": master_seed,
        "task": "temporal thread-mutex graph state classification and future deadlock prediction",
        "class_vocabulary": list(LABELS),
        "scenario_vocabulary": list(SCENARIOS),
        "snapshot_interval_ms": 10,
        "sequence_length": SEQUENCE_LENGTH,
        "sequence_stride": {"train": 4, "validation": 8, "test": 8},
        "confirmation_ms": CONFIRM_STEPS * 10,
        "prediction_horizon_ms": PREDICTION_HORIZON_STEPS * 10,
        "input_exclusions": [
            "label", "label_metadata", "has_cycle", "cycle_nodes", "targets",
            "provenance", "run_id", "snapshot_id", "graph_features",
        ],
        "split_policy": (
            "run-disjoint and seed-disjoint; validation/test use shifted larger-graph "
            "parameter distributions and non-overlapping windows"
        ),
        "splits": {},
    }
    for split in ("train", "validation", "test"):
        label_counts = Counter(row["label"] for row in split_sequences[split])
        scenario_counts = Counter(row["provenance"]["scenario"] for row in split_sequences[split])
        metadata["splits"][split] = {
            "run_count": sum(row["split"] == split for row in manifest),
            "snapshot_count": len(split_snapshots[split]),
            "sequence_count": len(split_sequences[split]),
            "sequence_labels": dict(sorted(label_counts.items())),
            "scenario_sequences": dict(sorted(scenario_counts.items())),
        }

    train_counts = metadata["splits"]["train"]["sequence_labels"]
    train_total = sum(train_counts.values())
    metadata["recommended_class_weights"] = {
        label: train_total / (len(LABELS) * train_counts.get(label, 1))
        for label in LABELS
    }

    topology_labels: dict[str, set[str]] = defaultdict(set)
    for snapshot in split_snapshots["validation"]:
        topology_labels[topology_signature(snapshot)].add(snapshot["label"])
    metadata["validation_multi_label_topologies"] = sum(
        len(labels) > 1 for labels in topology_labels.values()
    )
    (output / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("dataset/v3"))
    parser.add_argument("--runs-per-scenario", type=int, default=240)
    parser.add_argument("--seed", type=int, default=20260830)
    args = parser.parse_args()
    metadata = generate(args.output, args.runs_per_scenario, args.seed)
    print(json.dumps(metadata["splits"], indent=2))


if __name__ == "__main__":
    main()
