from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .graph import build_snapshots, read_events, write_jsonl
from .validation import validate_snapshots


@dataclass(frozen=True)
class TemporalRun:
    run_id: str
    run_dir: Path
    scenario: str
    mode: str
    noise: str
    threads: int
    seed: int
    iterations: int
    timeout_ms: int
    kernel: str
    machine: str
    workload_status: int
    snapshots: tuple[dict[str, Any], ...]
    sensor_counts: dict[str, int]


def _expected_threads(scenario: str, thread_counts: list[int]) -> list[int]:
    return [2] if scenario == "abba" else thread_counts


def expected_keys(config: dict[str, Any]) -> set[tuple[str, str, int, int, str]]:
    return {
        (scenario, mode, threads, seed, noise)
        for scenario in config["scenarios"]
        for mode in config["modes"]
        for threads in _expected_threads(scenario, config["thread_counts"])
        for seed in config["seeds"]
        for noise in config["noise"]
    }


def _sensor_errors(counts: Counter[str], mode: str) -> list[str]:
    errors = []
    required = {"lock_attempt", "lock_acquired", "lock_released", "sched_switch"}
    if mode == "deadlock":
        required.add("futex_wait")
    for event in sorted(required):
        if counts[event] == 0:
            errors.append(f"missing required eBPF sensor event: {event}")
    return errors


def build_run(run_dir: Path, config: dict[str, Any]) -> tuple[TemporalRun | None, list[str]]:
    summary_path = run_dir / "run_summary.json"
    workload_path = run_dir / "workload.jsonl"
    ebpf_path = run_dir / "ebpf.jsonl"
    if not all(path.exists() for path in (summary_path, workload_path, ebpf_path)):
        return None, ["missing raw run artifact"]

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    events = read_events(workload_path) + read_events(ebpf_path)
    events.sort(key=lambda event: (event["ts_ns"], event.get("seq", 0)))
    sensor_counts = Counter(
        event["event"] for event in events if event.get("source") == "ebpf"
    )
    snapshots = build_snapshots(
        events,
        interval_ns=int(config["snapshot_interval_ms"]) * 1_000_000,
        unsafe_window_ns=int(config["unsafe_window_ms"]) * 1_000_000,
        confirm_ns=int(config["confirmation_ms"]) * 1_000_000,
        graph_features={},
        start_policy="first_mutex_event",
        event_aligned=False,
        post_cycle_ns=int(config["post_cycle_ms"]) * 1_000_000,
        drop_empty=True,
        drop_edgeless=False,
        deduplicate=False,
    )
    if snapshots:
        interval_ns = int(config["snapshot_interval_ms"]) * 1_000_000
        grid_start = snapshots[0]["ts_ns"]
        snapshots = [
            snapshot for snapshot in snapshots
            if (snapshot["ts_ns"] - grid_start) % interval_ns == 0
        ]

    for index, snapshot in enumerate(snapshots):
        if snapshot["label"] == "unsafe":
            snapshot["label"] = "pre_deadlock"
        snapshot["snapshot_index"] = index
        snapshot["snapshot_id"] = f"{summary.get('run_id', run_dir.name)}:{index:05d}"
        snapshot["delta_ns"] = 0 if index == 0 else (
            snapshot["ts_ns"] - snapshots[index - 1]["ts_ns"]
        )
        snapshot["label_metadata"]["class_vocabulary"] = [
            "safe", "pre_deadlock", "deadlocked"
        ]

    validation_view = [
        {**snapshot, "label": "unsafe" if snapshot["label"] == "pre_deadlock" else snapshot["label"]}
        for snapshot in snapshots
    ]
    errors = validate_snapshots(validation_view, summary.get("mode"), require_active=False)
    errors.extend(_sensor_errors(sensor_counts, str(summary.get("mode"))))
    labels = Counter(snapshot["label"] for snapshot in snapshots)
    if len(snapshots) < int(config["sequence_length"]):
        errors.append("run is shorter than one temporal sequence")
    if summary.get("mode") == "safe" and set(labels) != {"safe"}:
        errors.append(f"safe run has unexpected labels: {dict(labels)}")
    if summary.get("mode") == "deadlock":
        missing = {"safe", "pre_deadlock", "deadlocked"} - set(labels)
        if missing:
            errors.append(f"deadlock run is missing temporal classes: {sorted(missing)}")
        first_cycle = next((s for s in snapshots if s["has_cycle"]), None)
        if first_cycle is None:
            errors.append("deadlock run has no observed cycle")
        elif not any(
            s["label"] == "pre_deadlock" and not s["has_cycle"] for s in snapshots
        ):
            errors.append("deadlock run has no pre-cycle pre_deadlock snapshot")
    if errors:
        return None, errors

    return TemporalRun(
        run_id=str(summary.get("run_id", run_dir.name)),
        run_dir=run_dir,
        scenario=str(summary["scenario"]),
        mode=str(summary["mode"]),
        noise=str(summary["noise"]),
        threads=int(summary["threads"]),
        seed=int(summary["seed"]),
        iterations=int(summary.get("iterations", 0)),
        timeout_ms=int(summary["timeout_ms"]),
        kernel=str(summary.get("kernel", "unknown")),
        machine=str(summary.get("machine", "unknown")),
        workload_status=int(summary.get("workload_status", 0)),
        snapshots=tuple(snapshots),
        sensor_counts=dict(sensor_counts),
    ), []


def discover_temporal_runs(
    runs_dir: Path, config: dict[str, Any]
) -> tuple[list[TemporalRun], list[dict[str, Any]]]:
    wanted = expected_keys(config)
    candidates: dict[tuple[str, str, int, int, str], list[TemporalRun]] = defaultdict(list)
    rejected: list[dict[str, Any]] = []
    for run_dir in sorted(path for path in runs_dir.iterdir() if path.is_dir()):
        summary_path = run_dir / "run_summary.json"
        if not summary_path.exists():
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        key = (
            str(summary.get("scenario")), str(summary.get("mode")),
            int(summary.get("threads", 0)), int(summary.get("seed", -1)),
            str(summary.get("noise")),
        )
        if (
            key not in wanted
            or int(summary.get("iterations", 0)) != int(config["iterations"])
            or summary.get("generation_id") != config.get("generation_id")
        ):
            continue
        run, errors = build_run(run_dir, config)
        if run is None:
            rejected.append({"run_id": run_dir.name, "errors": errors})
        else:
            candidates[key].append(run)

    selected = []
    for key in sorted(candidates):
        group = sorted(candidates[key], key=lambda run: run.run_id)
        selected.append(group[-1])
        for duplicate in group[:-1]:
            rejected.append({"run_id": duplicate.run_id, "errors": ["superseded duplicate"]})
    return selected, rejected


def split_by_seed(runs: list[TemporalRun], seeds: list[int]) -> dict[str, list[TemporalRun]]:
    ordered = sorted(seeds)
    train_count = max(1, int(len(ordered) * 0.70))
    validation_count = max(1, int(len(ordered) * 0.15))
    if train_count + validation_count >= len(ordered):
        validation_count = 1
        train_count = len(ordered) - 2
    seed_splits = {
        "train": set(ordered[:train_count]),
        "validation": set(ordered[train_count:train_count + validation_count]),
        "test": set(ordered[train_count + validation_count:]),
    }
    return {
        split: sorted(
            (run for run in runs if run.seed in split_seeds),
            key=lambda run: run.run_id,
        )
        for split, split_seeds in seed_splits.items()
    }


def make_sequences(
    run: TemporalRun, length: int, stride: int
) -> list[dict[str, Any]]:
    sequences = []
    snapshots = run.snapshots
    for end in range(length - 1, len(snapshots), stride):
        start = end - length + 1
        window = snapshots[start:end + 1]
        sequences.append({
            "sequence_id": f"{run.run_id}:{start:05d}-{end:05d}",
            "run_id": run.run_id,
            "snapshot_ids": [snapshot["snapshot_id"] for snapshot in window],
            "start_ts_ns": window[0]["ts_ns"],
            "end_ts_ns": window[-1]["ts_ns"],
            "duration_ns": window[-1]["ts_ns"] - window[0]["ts_ns"],
            "label": window[-1]["label"],
            "has_cycle": window[-1]["has_cycle"],
            "provenance": {
                "scenario": run.scenario,
                "mode": run.mode,
                "noise": run.noise,
                "threads": run.threads,
                "seed": run.seed,
            },
        })
    return sequences


def write_temporal_dataset(
    output_dir: Path, runs: list[TemporalRun], rejected: list[dict[str, Any]],
    config: dict[str, Any], config_path: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    splits = split_by_seed(runs, list(config["seeds"]))
    sequence_length = int(config["sequence_length"])
    sequence_stride = int(config["sequence_stride"])
    manifest = []
    metadata: dict[str, Any] = {
        "dataset_version": output_dir.name,
        "task": "temporal synchronization graph classification",
        "class_vocabulary": ["safe", "pre_deadlock", "deadlocked"],
        "config": config,
        "config_source": str(config_path),
        "split_policy": "globally seed-disjoint 70/10/20 split",
        "splits": {},
        "curation": {
            "expected_configurations": len(expected_keys(config)),
            "selected_runs": len(runs),
            "rejected_runs": rejected,
        },
    }
    for split, split_runs in splits.items():
        snapshots = [snapshot for run in split_runs for snapshot in run.snapshots]
        sequences = [
            sequence for run in split_runs
            for sequence in make_sequences(run, sequence_length, sequence_stride)
        ]
        write_jsonl(output_dir / f"{split}.jsonl", snapshots)
        write_jsonl(output_dir / f"{split}_sequences.jsonl", sequences)
        snapshot_labels = Counter(snapshot["label"] for snapshot in snapshots)
        sequence_labels = Counter(sequence["label"] for sequence in sequences)
        metadata["splits"][split] = {
            "run_count": len(split_runs),
            "snapshot_count": len(snapshots),
            "sequence_count": len(sequences),
            "seeds": sorted({run.seed for run in split_runs}),
            "snapshot_labels": dict(sorted(snapshot_labels.items())),
            "sequence_labels": dict(sorted(sequence_labels.items())),
            "runs": [run.run_id for run in split_runs],
        }
        manifest.extend({
            "run_id": run.run_id,
            "dataset_version": output_dir.name,
            "split": split,
            "scenario": run.scenario,
            "mode": run.mode,
            "noise": run.noise,
            "threads": run.threads,
            "seed": run.seed,
            "iterations": run.iterations,
            "timeout_ms": run.timeout_ms,
            "kernel": run.kernel,
            "machine": run.machine,
            "workload_status": run.workload_status,
            "snapshot_count": len(run.snapshots),
            "sequence_count": len(make_sequences(run, sequence_length, sequence_stride)),
            "sensor_counts": run.sensor_counts,
        } for run in split_runs)

    train_counts = metadata["splits"]["train"]["sequence_labels"]
    total_train = sum(train_counts.values())
    metadata["recommended_class_weights"] = {
        label: total_train / (len(train_counts) * count)
        for label, count in train_counts.items()
    }
    write_jsonl(output_dir / "run_manifest.jsonl", sorted(manifest, key=lambda x: x["run_id"]))
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return metadata


def build_dataset(
    runs_dir: Path, output_dir: Path, config_path: Path
) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    runs, rejected = discover_temporal_runs(runs_dir, config)
    for run in runs:
        write_jsonl(run.run_dir / "snapshots-v2.jsonl", run.snapshots)
    return write_temporal_dataset(output_dir, runs, rejected, config, config_path)
