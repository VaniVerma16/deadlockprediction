from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .graph import read_events
from .validation import validate_snapshots


CLASSES = {"safe", "pre_deadlock", "deadlocked"}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _finite_features(snapshot: dict[str, Any]) -> bool:
    for node in snapshot.get("nodes", []):
        for value in node.get("features", {}).values():
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                return False
    for value in snapshot.get("graph_features", {}).values():
        if value is not None and (
            not isinstance(value, (int, float)) or not math.isfinite(value)
        ):
            return False
    return True


def validate_temporal_dataset(dataset_dir: Path) -> dict[str, Any]:
    metadata = json.loads((dataset_dir / "metadata.json").read_text(encoding="utf-8"))
    manifest = _read_jsonl(dataset_dir / "run_manifest.jsonl")
    errors: list[str] = []
    warnings: list[str] = []
    report: dict[str, Any] = {"dataset": str(dataset_dir), "splits": {}}
    sequence_length = int(metadata["config"]["sequence_length"])
    expected_interval = int(metadata["config"]["snapshot_interval_ms"]) * 1_000_000

    manifest_runs: dict[str, dict[str, Any]] = {}
    split_runs: dict[str, set[str]] = defaultdict(set)
    split_seeds: dict[str, set[int]] = defaultdict(set)
    for record in manifest:
        run_id = record["run_id"]
        if run_id in manifest_runs:
            errors.append(f"duplicate manifest run_id: {run_id}")
        manifest_runs[run_id] = record
        split_runs[record["split"]].add(run_id)
        split_seeds[record["split"]].add(int(record["seed"]))
        required_sensors = {"lock_attempt", "lock_acquired", "lock_released", "sched_switch"}
        if record["mode"] == "deadlock":
            required_sensors.add("futex_wait")
        missing_sensors = [
            event for event in required_sensors
            if int(record.get("sensor_counts", {}).get(event, 0)) == 0
        ]
        if missing_sensors:
            errors.append(f"{run_id}: missing sensors {missing_sensors}")

    split_names = ["train", "validation", "test"]
    for left_index, left in enumerate(split_names):
        for right in split_names[left_index + 1:]:
            overlap = split_runs[left] & split_runs[right]
            if overlap:
                errors.append(f"run leakage between {left} and {right}: {sorted(overlap)[:3]}")
            seed_overlap = split_seeds[left] & split_seeds[right]
            if seed_overlap:
                errors.append(f"seed leakage between {left} and {right}: {sorted(seed_overlap)}")

    all_snapshot_ids: set[str] = set()
    all_sequence_ids: set[str] = set()
    for split in split_names:
        snapshots = read_events(dataset_dir / f"{split}.jsonl")
        sequences = _read_jsonl(dataset_dir / f"{split}_sequences.jsonl")
        snapshot_by_id: dict[str, dict[str, Any]] = {}
        by_run: dict[str, list[dict[str, Any]]] = defaultdict(list)
        label_counts = Counter()
        topology_signatures = Counter()
        for snapshot in snapshots:
            snapshot_id = snapshot.get("snapshot_id")
            if not isinstance(snapshot_id, str) or not snapshot_id:
                errors.append(f"{split}: snapshot without snapshot_id")
                continue
            if snapshot_id in all_snapshot_ids:
                errors.append(f"duplicate snapshot_id across dataset: {snapshot_id}")
            all_snapshot_ids.add(snapshot_id)
            snapshot_by_id[snapshot_id] = snapshot
            by_run[snapshot["run_id"]].append(snapshot)
            label_counts[snapshot["label"]] += 1
            if not _finite_features(snapshot):
                errors.append(f"{snapshot_id}: non-finite or non-numeric feature")
            topology_signatures[json.dumps({
                "nodes": [(node["type"], node["features"].get("is_waiting", 0),
                           node["features"].get("has_owner", 0)) for node in snapshot["nodes"]],
                "edges": [(edge["source"].split(":", 1)[0], edge["target"].split(":", 1)[0], edge["type"])
                          for edge in snapshot["edges"]],
                "label": snapshot["label"],
            }, sort_keys=True)] += 1

        validation_view = [
            {**snapshot, "label": "unsafe" if snapshot["label"] == "pre_deadlock" else snapshot["label"]}
            for snapshot in snapshots
        ]
        graph_errors = validate_snapshots(validation_view)
        errors.extend(f"{split}: {error}" for error in graph_errors)

        for run_id, run_snapshots in by_run.items():
            run_snapshots.sort(key=lambda item: item["snapshot_index"])
            expected_indices = list(range(len(run_snapshots)))
            actual_indices = [item["snapshot_index"] for item in run_snapshots]
            if actual_indices != expected_indices:
                errors.append(f"{run_id}: non-contiguous snapshot indices")
            for index, snapshot in enumerate(run_snapshots):
                expected_delta = 0 if index == 0 else expected_interval
                if snapshot["delta_ns"] != expected_delta:
                    errors.append(
                        f"{snapshot['snapshot_id']}: delta_ns={snapshot['delta_ns']} "
                        f"expected={expected_delta}"
                    )
            mode = manifest_runs.get(run_id, {}).get("mode")
            run_labels = {snapshot["label"] for snapshot in run_snapshots}
            if mode == "safe" and run_labels != {"safe"}:
                errors.append(f"{run_id}: safe run labels are {sorted(run_labels)}")
            if mode == "deadlock" and run_labels != CLASSES:
                errors.append(f"{run_id}: deadlock run labels are {sorted(run_labels)}")
            if mode == "deadlock" and not any(
                snapshot["label"] == "pre_deadlock" and not snapshot["has_cycle"]
                for snapshot in run_snapshots
            ):
                errors.append(f"{run_id}: no true pre-cycle early-warning state")

        sequence_labels = Counter()
        for sequence in sequences:
            sequence_id = sequence.get("sequence_id")
            if sequence_id in all_sequence_ids:
                errors.append(f"duplicate sequence_id: {sequence_id}")
            all_sequence_ids.add(sequence_id)
            ids = sequence.get("snapshot_ids", [])
            if len(ids) != sequence_length:
                errors.append(f"{sequence_id}: sequence length {len(ids)} != {sequence_length}")
                continue
            missing = [snapshot_id for snapshot_id in ids if snapshot_id not in snapshot_by_id]
            if missing:
                errors.append(f"{sequence_id}: missing snapshot references {missing[:2]}")
                continue
            window = [snapshot_by_id[snapshot_id] for snapshot_id in ids]
            if len({snapshot["run_id"] for snapshot in window}) != 1:
                errors.append(f"{sequence_id}: sequence crosses run boundary")
            indices = [snapshot["snapshot_index"] for snapshot in window]
            if indices != list(range(indices[0], indices[0] + sequence_length)):
                errors.append(f"{sequence_id}: sequence snapshots are not consecutive")
            if sequence["label"] != window[-1]["label"]:
                errors.append(f"{sequence_id}: target label does not match final snapshot")
            if sequence["has_cycle"] != window[-1]["has_cycle"]:
                errors.append(f"{sequence_id}: cycle target does not match final snapshot")
            sequence_labels[sequence["label"]] += 1

        missing_classes = CLASSES - set(sequence_labels)
        if missing_classes:
            errors.append(f"{split}: sequence classes missing {sorted(missing_classes)}")
        report["splits"][split] = {
            "runs": len(by_run),
            "seeds": sorted(split_seeds[split]),
            "snapshots": len(snapshots),
            "sequences": len(sequences),
            "snapshot_labels": dict(sorted(label_counts.items())),
            "sequence_labels": dict(sorted(sequence_labels.items())),
            "unique_topology_label_signatures": len(topology_signatures),
            "largest_topology_label_signature_fraction": (
                max(topology_signatures.values()) / len(snapshots) if snapshots else 0
            ),
        }

    expected_runs = int(metadata["curation"]["expected_configurations"])
    if len(manifest_runs) != expected_runs:
        errors.append(f"manifest has {len(manifest_runs)} runs; expected {expected_runs}")
    report["totals"] = {
        "runs": len(manifest_runs),
        "snapshots": len(all_snapshot_ids),
        "sequences": len(all_sequence_ids),
    }
    report["errors"] = errors
    report["warnings"] = warnings
    report["valid"] = not errors
    return report


def write_validation_report(dataset_dir: Path) -> dict[str, Any]:
    report = validate_temporal_dataset(dataset_dir)
    (dataset_dir / "validation_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report
