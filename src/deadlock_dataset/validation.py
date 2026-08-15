from __future__ import annotations

from collections import Counter
from typing import Any


VALID_LABELS = {"safe", "unsafe", "pre_deadlock", "deadlocked"}


def validate_snapshots(
    snapshots: list[dict[str, Any]], expected_mode: str | None = None,
    require_active: bool = False,
) -> list[str]:
    errors: list[str] = []
    previous_ts: dict[str, int] = {}
    for index, snapshot in enumerate(snapshots):
        prefix = f"record {index}"
        run_id = snapshot.get("run_id")
        timestamp = snapshot.get("ts_ns")
        if not isinstance(run_id, str) or not run_id:
            errors.append(f"{prefix}: missing run_id")
        if not isinstance(timestamp, int) or timestamp < 0:
            errors.append(f"{prefix}: invalid ts_ns")
            continue
        if run_id in previous_ts and timestamp <= previous_ts[run_id]:
            errors.append(f"{prefix}: timestamps are not strictly increasing")
        previous_ts[run_id] = timestamp

        label = snapshot.get("label")
        if label not in VALID_LABELS:
            errors.append(f"{prefix}: invalid label {label!r}")
        if label == "deadlocked" and not snapshot.get("has_cycle"):
            errors.append(f"{prefix}: deadlocked label has no cycle")

        nodes = snapshot.get("nodes", [])
        edges = snapshot.get("edges", [])
        node_types = {node.get("id"): node.get("type") for node in nodes}
        node_ids = set(node_types)
        if None in node_ids:
            errors.append(f"{prefix}: node without id")
        if require_active and (not nodes or not edges):
            errors.append(f"{prefix}: active dataset record is empty or edgeless")
        for edge in edges:
            if edge.get("source") not in node_ids or edge.get("target") not in node_ids:
                errors.append(f"{prefix}: edge references an unknown node")
                continue
            edge_type = edge.get("type")
            source_type = node_types[edge["source"]]
            target_type = node_types[edge["target"]]
            if edge_type == "owned_by" and (source_type, target_type) != ("lock", "thread"):
                errors.append(f"{prefix}: owned_by edge has the wrong direction")
            elif edge_type == "waits_for" and (source_type, target_type) != ("thread", "lock"):
                errors.append(f"{prefix}: waits_for edge has the wrong direction")
            elif edge_type not in {"owned_by", "waits_for"}:
                errors.append(f"{prefix}: invalid edge type {edge_type!r}")
        cycle_nodes = snapshot.get("cycle_nodes", [])
        if bool(cycle_nodes) != bool(snapshot.get("has_cycle")):
            errors.append(f"{prefix}: has_cycle disagrees with cycle_nodes")
        if not set(cycle_nodes).issubset(node_ids):
            errors.append(f"{prefix}: cycle references an unknown node")
        if label == "safe" and snapshot.get("has_cycle"):
            errors.append(f"{prefix}: safe label contains a cycle")
    labels = {snapshot.get("label") for snapshot in snapshots}
    if expected_mode == "deadlock" and "deadlocked" not in labels:
        errors.append("run expected a deadlock but no deadlocked snapshot was observed")
    if expected_mode == "safe" and "deadlocked" in labels:
        errors.append("safe run contains a deadlocked snapshot")
    return errors


def dataset_summary(snapshots: list[dict[str, Any]]) -> dict[str, Any]:
    labels = Counter(snapshot.get("label", "missing") for snapshot in snapshots)
    return {
        "records": len(snapshots),
        "runs": len({snapshot.get("run_id") for snapshot in snapshots}),
        "labels": dict(sorted(labels.items())),
    }
