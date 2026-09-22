from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from v5_inference import V5Inference


def run(sequence_path: str) -> None:
    sequence_file = Path(sequence_path)

    if not sequence_file.exists():
        raise FileNotFoundError(
            f"Sequence file not found: {sequence_file}"
        )

    print("=" * 80)
    print("V5 TEMPORAL SEQUENCE INFERENCE")
    print("=" * 80)
    print(f"Sequence: {sequence_file}")
    print()

    inference = V5Inference()

    results = []

    with sequence_file.open("r", encoding="utf-8") as f:
        for index, line in enumerate(f, start=1):
            line = line.strip()

            if not line:
                continue

            snapshot = json.loads(line)

            result = inference.add_snapshot(snapshot)

            if not result["ready"]:
                print(
                    f"Snapshot {index:02d}: "
                    f"warming up "
                    f"({result['snapshots']}/{result['required']})"
                )
                continue

            results.append(result)

            print(
                f"Snapshot {index:02d}: "
                f"state={result['state']:<12} "
                f"deadlock={result['deadlock_probability']:.4f} "
                f"pre={result['pre_deadlock_probability']:.4f} "
                f"risk50={result['risk_50ms']:.4f} "
                f"risk100={result['risk_100ms']:.4f} "
                f"risk300={result['risk_300ms']:.4f}"
            )

    print()
    print("=" * 80)

    if not results:
        print("No prediction produced.")
        print("At least 8 snapshots are required.")
        return

    final = results[-1]

    print(f"FINAL_STATE={final['state']}")
    print(
        f"FINAL_DEADLOCK_PROBABILITY="
        f"{final['deadlock_probability']:.4f}"
    )
    print(
        f"FINAL_PRE_DEADLOCK_PROBABILITY="
        f"{final['pre_deadlock_probability']:.4f}"
    )
    print(
        f"FINAL_RISK_50MS="
        f"{final['risk_50ms']:.4f}"
    )
    print(
        f"FINAL_RISK_100MS="
        f"{final['risk_100ms']:.4f}"
    )
    print(
        f"FINAL_RISK_300MS="
        f"{final['risk_300ms']:.4f}"
    )


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(
            "Usage: python experiments/run_v5_sequence.py "
            "<sequence.jsonl>"
        )
        raise SystemExit(1)

    run(sys.argv[1])