from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from deadlock_prototype.inference import classify_snapshot
from deadlock_prototype.schema import validate_snapshot


VALID_LABELS = {
    "safe",
    "pre_deadlock",
    "deadlocked",
}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run baseline inference on a converted graph snapshot"
    )

    parser.add_argument(
        "--snapshot",
        type=Path,
        required=True,
        help="Converted graph snapshot JSON",
    )

    args = parser.parse_args()

    snapshot_payload = json.loads(
        args.snapshot.read_text(
            encoding="utf-8"
        )
    )

    snapshot = validate_snapshot(
        snapshot_payload
    )

    # IMPORTANT:
    # This stage accepts converted graph data only.
    # It never reads raw runtime events.
    state = classify_snapshot(snapshot)

    if state not in VALID_LABELS:
        raise RuntimeError(
            f"unexpected inference state: {state!r}"
        )

    print(state)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())