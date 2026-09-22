from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable


VALID_LABELS = {
    "safe",
    "pre_deadlock",
    "deadlocked",
}


def run_command(command: list[str]) -> None:
    """Run a pipeline stage and stop immediately if it fails."""

    subprocess.run(
        command,
        check=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run workload -> raw events -> graph sequence -> "
            "baseline inference"
        )
    )

    parser.add_argument(
        "--mode",
        choices=[
            "safe",
            "contention",
            "deadlock",
        ],
        default="safe",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "snapshots",
    )

    args = parser.parse_args()

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    raw = (
        args.output_dir
        / f"{args.mode}-events.jsonl"
    )

    snapshot = (
        args.output_dir
        / f"{args.mode}-snapshot.json"
    )

    sequence = (
        args.output_dir
        / f"{args.mode}-sequence.jsonl"
    )

    # ---------------------------------------------------------
    # STAGE 1
    # Workload -> raw runtime events
    # ---------------------------------------------------------

    print(
        "\n[1/3] Running workload and collecting raw events..."
    )

    run_command(
        [
            PYTHON,
            str(ROOT / "experiments" / "run_demo.py"),
            "--mode",
            args.mode,
            "--output-dir",
            str(args.output_dir),
        ]
    )

    if not raw.exists():
        raise RuntimeError(
            f"raw event file was not created: {raw}"
        )

    # ---------------------------------------------------------
    # STAGE 2
    # Raw events -> graph snapshots
    # ---------------------------------------------------------

    print(
        "\n[2/3] Converting raw events into graph snapshots..."
    )

    run_command(
        [
            PYTHON,
            str(ROOT / "experiments" / "convert_events.py"),
            "--events",
            str(raw),
            "--snapshot",
            str(snapshot),
            "--sequence",
            str(sequence),
        ]
    )

    if not snapshot.exists():
        raise RuntimeError(
            f"snapshot file was not created: {snapshot}"
        )

    if not sequence.exists():
        raise RuntimeError(
            f"sequence file was not created: {sequence}"
        )

    # ---------------------------------------------------------
    # STAGE 3
    # Converted graph snapshot -> baseline inference
    # ---------------------------------------------------------

    print(
        "\n[3/3] Running baseline inference..."
    )

    result = subprocess.run(
        [
            PYTHON,
            str(ROOT / "experiments" / "infer_snapshot.py"),
            "--snapshot",
            str(snapshot),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    state = result.stdout.strip()

    if state not in VALID_LABELS:
        raise RuntimeError(
            f"unexpected final state: {state!r}"
        )

    print(
        f"\nFINAL_STATE={state}"
    )

    print(
        f"RAW_EVENTS={raw}"
    )

    print(
        f"GRAPH_SEQUENCE={sequence}"
    )

    print(
        f"INSPECTION_SNAPSHOT={snapshot}"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())