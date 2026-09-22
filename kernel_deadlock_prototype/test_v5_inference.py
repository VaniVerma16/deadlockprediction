from pathlib import Path
import json

from v5_inference import V5Inference


SNAPSHOT_DIR = (
    Path(__file__).resolve().parent
    / "snapshots"
)

SEQUENCES = [
    ("SAFE", SNAPSHOT_DIR / "safe-sequence.jsonl"),
    ("CONTENTION", SNAPSHOT_DIR / "contention-sequence.jsonl"),
    ("DEADLOCK", SNAPSHOT_DIR / "deadlock-sequence.jsonl"),
]


def load_snapshots(path: Path):

    snapshots = []

    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:

        for line in handle:

            line = line.strip()

            if line:
                snapshots.append(
                    json.loads(line)
                )

    return snapshots


def run_workload(
    model: V5Inference,
    name: str,
    path: Path,
):

    print()
    print("=" * 70)
    print(f"{name} WORKLOAD")
    print("=" * 70)

    snapshots = load_snapshots(path)

    print(
        f"Snapshots: {len(snapshots)}"
    )

    if len(snapshots) < 8:

        print(
            "Not enough snapshots for V5."
        )

        return

    # Reset the temporal window between workloads.
    model.window.clear()

    results = []

    for index, snapshot in enumerate(
        snapshots
    ):

        result = model.add_snapshot(
            snapshot
        )

        if not result["ready"]:
            continue

        results.append(
            (
                index + 1,
                result,
            )
        )

        print()
        print(
            f"Snapshot {index + 1}"
        )

        print(
            f"  State: "
            f"{result['state']}"
        )

        print(
            f"  P(deadlock): "
            f"{result['deadlock_probability']:.4f}"
        )

        print(
            f"  P(pre-deadlock): "
            f"{result['pre_deadlock_probability']:.4f}"
        )

        print(
            f"  Risk <=50ms: "
            f"{result['risk_50ms']:.4f}"
        )

        print(
            f"  Risk <=100ms: "
            f"{result['risk_100ms']:.4f}"
        )

        print(
            f"  Risk <=300ms: "
            f"{result['risk_300ms']:.4f}"
        )

    if results:

        final_index, final_result = results[-1]

        print()
        print("-" * 70)
        print("FINAL V5 RESULT")
        print("-" * 70)

        print(
            f"Final snapshot: "
            f"{final_index}"
        )

        print(
            f"Final state: "
            f"{final_result['state']}"
        )

        print(
            f"P(deadlock): "
            f"{final_result['deadlock_probability']:.4f}"
        )

        print(
            f"P(pre-deadlock): "
            f"{final_result['pre_deadlock_probability']:.4f}"
        )

        print(
            f"Risk <=50ms: "
            f"{final_result['risk_50ms']:.4f}"
        )

        print(
            f"Risk <=100ms: "
            f"{final_result['risk_100ms']:.4f}"
        )

        print(
            f"Risk <=300ms: "
            f"{final_result['risk_300ms']:.4f}"
        )


def main():

    print("=" * 70)
    print("V5 INFERENCE — ALL PROTOTYPE WORKLOADS")
    print("=" * 70)

    model = V5Inference()

    for name, path in SEQUENCES:

        if not path.exists():

            print(
                f"\nMissing: {path}"
            )

            continue

        run_workload(
            model,
            name,
            path,
        )


if __name__ == "__main__":
    main()