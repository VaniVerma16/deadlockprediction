from __future__ import annotations

import json
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .graph import read_events, write_jsonl
from .perf import read_perf_stat
from .validation import validate_snapshots


@dataclass(frozen=True)
class Run:
    run_id: str
    snapshots_path: Path
    scenario: str
    mode: str
    noise: str
    threads: int
    seed: int
    perf_valid: bool
    labels_valid: bool
    timeout_ms: int
    kernel: str
    machine: str
    workload_status: int


def discover_runs(runs_dir: Path, snapshot_name: str = "snapshots.jsonl") -> list[Run]:
    runs: list[Run] = []
    for snapshots_path in sorted(runs_dir.glob(f"*/{snapshot_name}")):
        run_dir = snapshots_path.parent
        summary_path = run_dir / "run_summary.json"
        summary: dict[str, Any] = {}
        if summary_path.exists():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        snapshots = read_events(snapshots_path)
        if not snapshots:
            continue
        provenance = snapshots[0].get("provenance", {})
        mode = summary.get("mode", provenance.get("mode", "unknown"))
        runs.append(Run(
            run_id=summary.get("run_id", snapshots[0]["run_id"]),
            snapshots_path=snapshots_path,
            scenario=summary.get("scenario", provenance.get("scenario", "unknown")),
            mode=mode,
            noise=summary.get("noise", "unknown"),
            threads=int(summary.get("threads", 0)),
            seed=int(summary.get("seed", provenance.get("seed", -1))),
            perf_valid=bool(read_perf_stat(run_dir / "perf.csv")),
            labels_valid=not validate_snapshots(snapshots, mode if mode in {"safe", "deadlock"} else None),
            timeout_ms=int(summary.get("timeout_ms", 0)),
            kernel=str(summary.get("kernel", "unknown")),
            machine=str(summary.get("machine", "unknown")),
            workload_status=int(summary.get("workload_status", 0)),
        ))
    return runs


def curate_runs(runs: list[Run]) -> tuple[list[Run], list[Run]]:
    candidates: dict[tuple[str, str, int, int, str], list[Run]] = defaultdict(list)
    excluded: list[Run] = []
    for run in runs:
        if not run.labels_valid:
            excluded.append(run)
            continue
        key = (run.scenario, run.mode, run.threads, run.seed, run.noise)
        candidates[key].append(run)

    selected: list[Run] = []
    for key in sorted(candidates):
        group = sorted(candidates[key], key=lambda run: (run.perf_valid, run.run_id))
        selected.append(group[-1])
        excluded.extend(group[:-1])
    return selected, excluded


def assign_splits(runs: list[Run], seed: int) -> dict[str, list[Run]]:
    randomizer = random.Random(seed)
    strata: dict[tuple[str, str, str, int], list[Run]] = defaultdict(list)
    for run in runs:
        strata[(run.scenario, run.mode, run.noise, run.threads)].append(run)

    splits: dict[str, list[Run]] = {"train": [], "validation": [], "test": []}
    for stratum in sorted(strata):
        group = sorted(strata[stratum], key=lambda item: item.run_id)
        randomizer.shuffle(group)
        count = len(group)
        if count >= 3:
            train_count = max(1, int(count * 0.70))
            validation_count = max(1, int(count * 0.15))
            if train_count + validation_count >= count:
                train_count = count - 2
                validation_count = 1
        elif count == 2:
            train_count, validation_count = 1, 1
        else:
            train_count, validation_count = 1, 0
        splits["train"].extend(group[:train_count])
        splits["validation"].extend(group[train_count:train_count + validation_count])
        splits["test"].extend(group[train_count + validation_count:])
    return splits


def write_splits(output_dir: Path, splits: dict[str, list[Run]], seed: int) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata: dict[str, Any] = {
        "dataset_version": output_dir.name,
        "seed": seed,
        "run_manifest": "run_manifest.jsonl",
        "splits": {},
    }
    manifest: list[dict[str, Any]] = []
    snapshot_sources: set[str] = set()
    for split_name, runs in splits.items():
        records = []
        for run in sorted(runs, key=lambda item: item.run_id):
            records.extend(read_events(run.snapshots_path))
            snapshot_sources.add(run.snapshots_path.name)
        write_jsonl(output_dir / f"{split_name}.jsonl", records)
        metadata["splits"][split_name] = {
            "runs": [run.run_id for run in sorted(runs, key=lambda item: item.run_id)],
            "run_count": len(runs),
            "snapshot_count": len(records),
        }
        if records and "build_policy" not in metadata:
            label_metadata = records[0].get("label_metadata", {})
            metadata["build_policy"] = {
                key: value for key, value in label_metadata.items()
                if key != "first_cycle_ts_ns"
            }
        manifest.extend({
            "run_id": run.run_id,
            "dataset_version": output_dir.name,
            "snapshot_source": run.snapshots_path.name,
            "split": split_name,
            "scenario": run.scenario,
            "mode": run.mode,
            "noise": run.noise,
            "threads": run.threads,
            "seed": run.seed,
            "timeout_ms": run.timeout_ms,
            "kernel": run.kernel,
            "machine": run.machine,
            "workload_status": run.workload_status,
            "perf_valid": run.perf_valid,
            "labels_valid": run.labels_valid,
        } for run in sorted(runs, key=lambda item: item.run_id))
    metadata["snapshot_sources"] = sorted(snapshot_sources)
    write_jsonl(output_dir / "run_manifest.jsonl", sorted(manifest, key=lambda item: item["run_id"]))
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return metadata
