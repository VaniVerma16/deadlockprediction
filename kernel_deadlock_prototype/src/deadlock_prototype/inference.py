from __future__ import annotations

from typing import Literal

from .schema import Snapshot

Label = Literal["safe", "pre_deadlock", "deadlocked"]


def classify_snapshot(snapshot: Snapshot) -> Label:
    """Small explainable baseline; replace this with GNN inference later."""

    rule_label = snapshot.metadata.get("rule_label")
    if rule_label in {"safe", "pre_deadlock", "deadlocked"}:
        return rule_label
    if snapshot.metadata.get("cycle_count", 0) > 0:
        return "deadlocked"
    if snapshot.metadata.get("max_wait_ns", 0) >= 50_000_000:
        return "pre_deadlock"
    return "safe"
