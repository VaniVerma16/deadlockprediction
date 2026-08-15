from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .graph import build_snapshots, read_events, write_jsonl
from .perf import read_perf_stat
from .split import assign_splits, curate_runs, discover_runs, write_splits
from .temporal import build_dataset as build_temporal_dataset
from .temporal_validation import write_validation_report
from .validation import dataset_summary, validate_snapshots


def _milliseconds(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("milliseconds must be non-negative")
    return parsed * 1_000_000


def build_command(args: argparse.Namespace) -> int:
    events = []
    for path in args.events:
        events.extend(read_events(path))
    events.sort(key=lambda event: (event["ts_ns"], event.get("seq", 0)))
    features = read_perf_stat(args.perf) if args.perf else {}
    snapshots = build_snapshots(
        events,
        args.interval_ms,
        args.unsafe_window_ms,
        args.confirm_ms,
        features,
        start_policy=args.start_policy,
        event_aligned=args.event_aligned,
        post_cycle_ns=args.post_cycle_ms,
        drop_empty=args.drop_empty,
        drop_edgeless=args.drop_edgeless,
        deduplicate=args.deduplicate,
    )
    write_jsonl(args.output, snapshots)
    print(json.dumps(dataset_summary(snapshots), indent=2, sort_keys=True))
    return 0


def validate_command(args: argparse.Namespace) -> int:
    snapshots = read_events(args.dataset)
    errors = validate_snapshots(snapshots, args.expected_mode, args.require_active)
    print(json.dumps(dataset_summary(snapshots), indent=2, sort_keys=True))
    if errors:
        for error in errors:
            print(f"error: {error}", file=sys.stderr)
        return 1
    print("dataset validation passed")
    return 0


def rebuild_command(args: argparse.Namespace) -> int:
    completed = 0
    failed: list[dict[str, object]] = []
    label_counts: dict[str, int] = {"safe": 0, "unsafe": 0, "deadlocked": 0}
    for run_dir in sorted(path for path in args.runs_dir.iterdir() if path.is_dir()):
        required = [run_dir / "workload.jsonl", run_dir / "ebpf.jsonl", run_dir / "run_summary.json"]
        if not all(path.exists() for path in required):
            continue
        summary = json.loads((run_dir / "run_summary.json").read_text(encoding="utf-8"))
        events = read_events(run_dir / "workload.jsonl") + read_events(run_dir / "ebpf.jsonl")
        events.sort(key=lambda event: (event["ts_ns"], event.get("seq", 0)))
        features = read_perf_stat(run_dir / "perf.csv")
        snapshots = build_snapshots(
            events,
            interval_ns=args.interval_ms,
            unsafe_window_ns=args.unsafe_window_ms,
            confirm_ns=args.confirm_ms,
            graph_features=features,
            start_policy="first_graph_edge",
            event_aligned=True,
            post_cycle_ns=args.post_cycle_ms,
            drop_empty=True,
            drop_edgeless=True,
            deduplicate=True,
        )
        errors = validate_snapshots(snapshots, summary.get("mode"), require_active=True)
        if not snapshots:
            errors.append("no active graph snapshots were produced")
        if any(not snapshot["nodes"] for snapshot in snapshots):
            errors.append("an empty graph survived active-window filtering")
        if any(not snapshot["edges"] for snapshot in snapshots):
            errors.append("an edgeless graph survived active-allocation filtering")
        if errors:
            failed.append({"run_id": summary.get("run_id", run_dir.name), "errors": errors})
            continue
        write_jsonl(run_dir / args.snapshot_name, snapshots)
        completed += 1
        for snapshot in snapshots:
            label_counts[snapshot["label"]] += 1

    report = {
        "completed_runs": completed,
        "failed_runs": failed,
        "snapshot_labels": label_counts,
        "snapshot_name": args.snapshot_name,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 1 if failed else 0


def split_command(args: argparse.Namespace) -> int:
    runs = discover_runs(args.runs_dir, args.snapshot_name)
    if not runs:
        print(f"no snapshots found under {args.runs_dir}", file=sys.stderr)
        return 1
    curated, excluded = curate_runs(runs)
    metadata = write_splits(args.output, assign_splits(curated, args.seed), args.seed)
    metadata["curation"] = {
        "discovered_runs": len(runs),
        "selected_runs": len(curated),
        "excluded_runs": [run.run_id for run in sorted(excluded, key=lambda run: run.run_id)],
    }
    (args.output / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))
    return 0


def temporal_build_command(args: argparse.Namespace) -> int:
    metadata = build_temporal_dataset(args.runs_dir, args.output, args.config)
    print(json.dumps({
        "dataset_version": metadata["dataset_version"],
        "curation": metadata["curation"],
        "splits": metadata["splits"],
    }, indent=2, sort_keys=True))
    expected = metadata["curation"]["expected_configurations"]
    selected = metadata["curation"]["selected_runs"]
    return 0 if selected == expected else 1


def temporal_validate_command(args: argparse.Namespace) -> int:
    report = write_validation_report(args.dataset)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["valid"] else 1


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="deadlock-dataset")
    commands = root.add_subparsers(dest="command", required=True)

    build = commands.add_parser("build", help="build labeled graph snapshots")
    build.add_argument("--events", type=Path, required=True, nargs="+")
    build.add_argument("--output", type=Path, required=True)
    build.add_argument("--interval-ms", type=_milliseconds, default=20_000_000)
    build.add_argument("--unsafe-window-ms", type=_milliseconds, default=250_000_000)
    build.add_argument("--confirm-ms", type=_milliseconds, default=50_000_000)
    build.add_argument("--perf", type=Path, help="optional perf stat CSV sidecar")
    build.add_argument(
        "--start-policy", choices=["run", "first_mutex_event", "first_graph_edge"], default="run"
    )
    build.add_argument("--event-aligned", action="store_true")
    build.add_argument("--post-cycle-ms", type=_milliseconds)
    build.add_argument("--drop-empty", action="store_true")
    build.add_argument("--drop-edgeless", action="store_true")
    build.add_argument("--deduplicate", action="store_true")
    build.set_defaults(handler=build_command)

    validate = commands.add_parser("validate", help="validate graph snapshots")
    validate.add_argument("--dataset", type=Path, required=True)
    validate.add_argument("--expected-mode", choices=["safe", "deadlock"])
    validate.add_argument("--require-active", action="store_true")
    validate.set_defaults(handler=validate_command)

    rebuild = commands.add_parser("rebuild", help="rebuild active-window snapshots from raw runs")
    rebuild.add_argument("--runs-dir", type=Path, required=True)
    rebuild.add_argument("--snapshot-name", default="snapshots-v1.jsonl")
    rebuild.add_argument("--interval-ms", type=_milliseconds, default=20_000_000)
    rebuild.add_argument("--unsafe-window-ms", type=_milliseconds, default=250_000_000)
    rebuild.add_argument("--confirm-ms", type=_milliseconds, default=50_000_000)
    rebuild.add_argument("--post-cycle-ms", type=_milliseconds, default=250_000_000)
    rebuild.set_defaults(handler=rebuild_command)

    split = commands.add_parser("split", help="make run-level train/validation/test splits")
    split.add_argument("--runs-dir", type=Path, required=True)
    split.add_argument("--output", type=Path, required=True)
    split.add_argument("--seed", type=int, default=20260725)
    split.add_argument("--snapshot-name", default="snapshots.jsonl")
    split.set_defaults(handler=split_command)

    temporal = commands.add_parser(
        "temporal-build", help="build seed-disjoint temporal graph snapshots and sequences"
    )
    temporal.add_argument("--runs-dir", type=Path, required=True)
    temporal.add_argument("--output", type=Path, required=True)
    temporal.add_argument("--config", type=Path, required=True)
    temporal.set_defaults(handler=temporal_build_command)

    temporal_validate = commands.add_parser(
        "temporal-validate", help="validate temporal graph and sequence integrity"
    )
    temporal_validate.add_argument("--dataset", type=Path, required=True)
    temporal_validate.set_defaults(handler=temporal_validate_command)
    return root


def main() -> int:
    args = parser().parse_args()
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
