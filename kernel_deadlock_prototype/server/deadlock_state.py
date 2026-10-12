"""Separate model risk predictions from graph-confirmed deadlocks.

A high deadlock probability is a warning, not proof that a cycle has formed.
The graph is a directed thread/lock dependency graph; a directed cycle is
evidence of a circular wait in the observed snapshot.
"""
from __future__ import annotations


def has_dependency_cycle(graph: dict) -> bool:
    adjacency: dict[str, list[str]] = {}
    for edge in graph.get("edges", []):
        if edge.get("type") not in {"owned_by", "waits_for"}:
            continue
        source, target = edge.get("source"), edge.get("target")
        if source is None or target is None:
            continue
        adjacency.setdefault(str(source), []).append(str(target))

    visited: set[str] = set()
    active: set[str] = set()

    def dfs(node: str) -> bool:
        visited.add(node)
        active.add(node)
        for neighbor in adjacency.get(node, []):
            if neighbor in active:
                return True
            if neighbor not in visited and dfs(neighbor):
                return True
        active.remove(node)
        return False

    return any(dfs(node) for node in adjacency if node not in visited)


def interpret_prediction(graph: dict, result: dict) -> dict:
    """Return a display interpretation without modifying raw model outputs."""
    observed_cycle = has_dependency_cycle(graph)
    model_state = result["state"]
    if observed_cycle:
        state = "deadlocked"
    elif model_state in {"deadlocked", "pre_deadlock"}:
        state = "pre_deadlock"
    else:
        state = "safe"

    return {
        "state": state,
        "model_state": model_state,
        "observed_cycle": observed_cycle,
        "state_source": "graph_cycle" if observed_cycle else "model_risk",
    }
