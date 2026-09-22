# Kernel Deadlock Prototype

A beginner-friendly, Python-only prototype that mirrors the repository's V3 graph style without requiring eBPF or privileged kernel tracing.

## What it does

1. Runs a multithreaded workload in safe, contention, or intentional-deadlock mode.
2. Logs thread IDs, lock waits, owners, releases, wait durations, and wakeups.
3. Builds a typed synchronization graph with `thread` and `lock` nodes.
4. Writes raw events and a validated JSON snapshot.
5. Returns `safe`, `pre_deadlock`, or `deadlocked` using an explainable rule baseline.

## Install on the Linux VM

```bash
cd kernel_deadlock_prototype
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Python-only demo modes:

```bash
python experiments/run_demo.py --mode safe
python experiments/run_demo.py --mode contention
python experiments/run_demo.py --mode deadlock
```

Outputs are written under `snapshots/`:

- `*-events.jsonl`: normalized runtime events
- `*-snapshot.json`: highest-risk graph snapshot with nodes, edges, features, and rule metadata
- `*-sequence.jsonl`: event-prefix snapshot sequence, preserving transient waits

## Libraries

- `json`, `dataclasses`, `typing`, `pathlib`, `time`, `threading`: standard-library collection and data handling.
- `pydantic`: validates the snapshot contract.
- `networkx`: builds the graph and detects cycles.
- `numpy`: creates a compact numeric feature vector.
- `torch` and `torch_geometric`: intentionally not required for this first prototype. Add a trained GNN after the event and snapshot contracts are stable.
- `perf`, `strace`, bpftrace, and BCC remain optional Linux observability integrations for a later collector.

The existing sibling repository `deadlockprediction/` contains the production-oriented eBPF collector and temporal dataset tooling. This directory is intentionally self-contained so the first demo can be understood and run without kernel privileges.
