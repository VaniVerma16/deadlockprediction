#!/usr/bin/env python3
"""Validate dataset/v3 and measure simple last-snapshot shortcuts."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path


LABELS = {"safe": 0, "pre_deadlock": 1, "deadlocked": 2}
SCENARIOS = {
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
}


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def feature_vector(snapshot: dict) -> list[float]:
    threads = [node for node in snapshot["nodes"] if node["type"] == "thread"]
    waits = [node["features"].get("wait_ns", 0) for node in threads]
    return [
        len(threads),
        sum(node["type"] == "lock" for node in snapshot["nodes"]),
        len(snapshot["edges"]),
        sum(edge["type"] == "waits_for" for edge in snapshot["edges"]),
        sum(edge["type"] == "owned_by" for edge in snapshot["edges"]),
        sum(node["features"].get("is_waiting", 0) for node in threads),
        math.log1p(max(waits, default=0)),
        math.log1p(sum(waits)),
        math.log1p(sum(node["features"].get("scheduler_switches", 0) for node in threads)),
        math.log1p(sum(node["features"].get("wakeups", 0) for node in threads)),
        sum(node["features"].get("cpu_migrations", 0) for node in threads),
        int(snapshot["has_cycle"]),
    ]


def load_baseline_data(root: Path, split: str):
    sequences = list(read_jsonl(root / f"{split}_sequences.jsonl"))
    labels_by_snapshot = {
        sequence["snapshot_ids"][-1]: LABELS[sequence["label"]]
        for sequence in sequences
    }
    features = {}
    for snapshot in read_jsonl(root / f"{split}.jsonl"):
        if snapshot["snapshot_id"] in labels_by_snapshot:
            features[snapshot["snapshot_id"]] = feature_vector(snapshot)
    ordered_ids = [sequence["snapshot_ids"][-1] for sequence in sequences]
    return (
        [features[snapshot_id] for snapshot_id in ordered_ids],
        [labels_by_snapshot[snapshot_id] for snapshot_id in ordered_ids],
    )


def metric_payload(y_true, y_pred) -> dict:
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1, 2]).tolist(),
    }


def shortcut_baselines(root: Path) -> dict:
    import numpy as np
    from sklearn.tree import DecisionTreeClassifier

    train_x, train_y = load_baseline_data(root, "train")
    val_x, val_y = load_baseline_data(root, "validation")
    train_x = np.asarray(train_x)
    train_y = np.asarray(train_y)
    val_x = np.asarray(val_x)
    val_y = np.asarray(val_y)

    results = {}
    variants = {
        "max_wait_only_tree": [6],
        "last_snapshot_tree_without_cycle": list(range(11)),
        "last_snapshot_tree_with_cycle": list(range(12)),
    }
    for name, columns in variants.items():
        model = DecisionTreeClassifier(
            max_depth=4,
            min_samples_leaf=30,
            class_weight="balanced",
            random_state=0,
        )
        model.fit(train_x[:, columns], train_y)
        results[name] = metric_payload(val_y, model.predict(val_x[:, columns]))

    cycle_prediction = np.where(val_x[:, -1] > 0, 2, 0)
    results["cycle_only_rule"] = metric_payload(val_y, cycle_prediction)
    results["note"] = (
        "Baselines use validation only. The test split is not used for model selection "
        "or shortcut measurement."
    )
    return results


def integrity(root: Path) -> dict:
    errors = []
    manifests = list(read_jsonl(root / "run_manifest.jsonl"))
    run_ids = [row["run_id"] for row in manifests]
    if len(run_ids) != len(set(run_ids)):
        errors.append("duplicate run_id in manifest")

    split_runs = {
        split: {row["run_id"] for row in manifests if row["split"] == split}
        for split in ("train", "validation", "test")
    }
    split_seeds = {
        split: {row["seed"] for row in manifests if row["split"] == split}
        for split in ("train", "validation", "test")
    }
    if any(split_runs[a] & split_runs[b] for a, b in (("train", "validation"), ("train", "test"), ("validation", "test"))):
        errors.append("run overlap between splits")
    if any(split_seeds[a] & split_seeds[b] for a, b in (("train", "validation"), ("train", "test"), ("validation", "test"))):
        errors.append("seed overlap between splits")

    recovered_runs = set()
    multiple_cycle_snapshots = 0
    cycle_safe_snapshots = 0
    new_wait_events = 0
    wait_age_violations = 0
    persistent_wait_age_violations = 0
    split_details = {}
    for split in ("train", "validation", "test"):
        snapshot_ids = set()
        snapshot_run = {}
        snapshot_index = {}
        labels_by_run: dict[str, list[str]] = defaultdict(list)
        snapshot_counts = Counter()
        previous_waits_by_run: dict[str, dict[str, tuple[str, int]]] = defaultdict(dict)

        for snapshot in read_jsonl(root / f"{split}.jsonl"):
            sid = snapshot["snapshot_id"]
            if sid in snapshot_ids:
                errors.append(f"duplicate snapshot id in {split}: {sid}")
                break
            snapshot_ids.add(sid)
            snapshot_run[sid] = snapshot["run_id"]
            snapshot_index[sid] = snapshot["snapshot_index"]
            snapshot_counts[snapshot["label"]] += 1
            labels_by_run[snapshot["run_id"]].append(snapshot["label"])

            nodes = {node["id"] for node in snapshot["nodes"]}
            if any(edge["source"] not in nodes or edge["target"] not in nodes for edge in snapshot["edges"]):
                errors.append(f"dangling edge in {sid}")
                break
            metadata = snapshot["label_metadata"]
            expected = (
                "deadlocked" if metadata["confirmed_cycle"]
                else "pre_deadlock" if metadata["steps_to_next_confirmation"] is not None
                else "safe"
            )
            if snapshot["label"] != expected:
                errors.append(f"incorrect causal label in {sid}")
                break
            multiple_cycle_snapshots += int(metadata["cycle_count"] > 1)
            cycle_safe_snapshots += int(snapshot["has_cycle"] and snapshot["label"] == "safe")

            wait_ns = {
                node["id"]: node["features"].get("wait_ns", 0)
                for node in snapshot["nodes"]
                if node["type"] == "thread"
            }
            current_waits = {
                edge["source"]: (edge["target"], wait_ns[edge["source"]])
                for edge in snapshot["edges"]
                if edge["type"] == "waits_for"
            }
            prior_waits = previous_waits_by_run[snapshot["run_id"]]
            for thread_id, (lock_id, age_ns) in current_waits.items():
                prior = prior_waits.get(thread_id)
                if prior is None or prior[0] != lock_id:
                    if snapshot["snapshot_index"] > 0:
                        new_wait_events += 1
                        if not 0 <= age_ns <= 10_000_000:
                            wait_age_violations += 1
                elif age_ns != prior[1] + 10_000_000:
                    persistent_wait_age_violations += 1
            previous_waits_by_run[snapshot["run_id"]] = current_waits

        sequence_counts = Counter()
        scenarios = set()
        previous_end: dict[str, int] = {}
        for sequence in read_jsonl(root / f"{split}_sequences.jsonl"):
            ids = sequence["snapshot_ids"]
            sequence_counts[sequence["label"]] += 1
            scenarios.add(sequence["provenance"]["scenario"])
            if len(ids) != 8 or any(sid not in snapshot_ids for sid in ids):
                errors.append(f"invalid sequence references in {sequence['sequence_id']}")
                break
            if len({snapshot_run[sid] for sid in ids}) != 1:
                errors.append(f"cross-run sequence: {sequence['sequence_id']}")
                break
            indices = [snapshot_index[sid] for sid in ids]
            if indices != list(range(indices[0], indices[0] + 8)):
                errors.append(f"non-consecutive sequence: {sequence['sequence_id']}")
                break
            if split != "train":
                prior = previous_end.get(sequence["run_id"], -1)
                if indices[0] <= prior:
                    errors.append(f"overlapping evaluation windows: {sequence['sequence_id']}")
                    break
                previous_end[sequence["run_id"]] = indices[-1]

        if scenarios != SCENARIOS:
            errors.append(f"missing scenarios in {split}: {sorted(SCENARIOS - scenarios)}")
        split_details[split] = {
            "runs": len(split_runs[split]),
            "snapshots": len(snapshot_ids),
            "sequences": sum(sequence_counts.values()),
            "snapshot_labels": dict(snapshot_counts),
            "sequence_labels": dict(sequence_counts),
            "scenarios": len(scenarios),
        }
        for run_id, run_labels in labels_by_run.items():
            if "deadlocked" in run_labels:
                last_deadlocked = max(i for i, label in enumerate(run_labels) if label == "deadlocked")
                if "safe" in run_labels[last_deadlocked + 1:]:
                    recovered_runs.add(run_id)

    if wait_age_violations:
        errors.append(f"{wait_age_violations} newly observed waits have impossible ages")
    if persistent_wait_age_violations:
        errors.append(
            f"{persistent_wait_age_violations} persistent waits have non-monotonic ages"
        )

    return {
        "passed": not errors,
        "errors": errors,
        "split_seed_counts": {split: len(seeds) for split, seeds in split_seeds.items()},
        "splits": split_details,
        "multiple_cycle_snapshots": multiple_cycle_snapshots,
        "safe_snapshots_containing_transient_cycles": cycle_safe_snapshots,
        "runs_with_confirmed_deadlock_then_safe_recovery": len(recovered_runs),
        "new_wait_events_checked": new_wait_events,
        "new_wait_age_violations": wait_age_violations,
        "persistent_wait_age_violations": persistent_wait_age_violations,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path("dataset/v3"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = {
        "dataset": str(args.dataset),
        "integrity": integrity(args.dataset),
        "shortcut_baselines": shortcut_baselines(args.dataset),
    }
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered)
    if not report["integrity"]["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
