from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from deadlock_prototype.events import RuntimeEvent
from deadlock_prototype.graph import build_snapshot_sequence, write_snapshot_json


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert raw logger JSONL into graph snapshots")
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--sequence", type=Path, required=True)
    args = parser.parse_args()

    events = [RuntimeEvent(**json.loads(line)) for line in args.events.read_text(encoding="utf-8").splitlines() if line]
    snapshots = build_snapshot_sequence(events)
    if not snapshots:
        raise RuntimeError("raw logger output contained no events")

    highest_risk = max(
        snapshots,
        key=lambda item: (
            item.metadata.get("cycle_count", 0),
            item.metadata.get("max_wait_ns", 0),
            item.ts_ns,
        ),
    )
    write_snapshot_json(highest_risk, args.snapshot)
    args.sequence.parent.mkdir(parents=True, exist_ok=True)
    with args.sequence.open("w", encoding="utf-8") as handle:
        for snapshot in snapshots:
            payload = snapshot.model_dump() if hasattr(snapshot, "model_dump") else snapshot.dict()
            handle.write(json.dumps(payload) + "\n")

    print(json.dumps({"snapshots": len(snapshots), "snapshot": str(args.snapshot), "sequence": str(args.sequence)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())