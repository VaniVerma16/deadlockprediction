#!/usr/bin/env python3

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from deadlock_prototype.digital_twin import SynchronizationDigitalTwin
from deadlock_prototype.events import RuntimeEvent


def main():
    twin = SynchronizationDigitalTwin()

    start = 1_000_000_000

    events = [
        RuntimeEvent(
            ts_ns=start,
            event="lock_acquired",
            tid=1,
            lock_id="lock:1:0xA",
            owner_tid=1,
        ),
        RuntimeEvent(
            ts_ns=start + 2_000_000,
            event="lock_wait_start",
            tid=2,
            lock_id="lock:1:0xA",
            owner_tid=1,
        ),
    ]

    for event in events:
        twin.update(event)

    snapshots = []
    for i in range(8):
        snapshot = twin.sample(f"test-{i:04d}")
        snapshots.append(snapshot)

    print("10-ms sampling validation:")
    for snapshot in snapshots:
        print(snapshot.snapshot_id, snapshot.ts_ns)

    differences = [
        b.ts_ns - a.ts_ns
        for a, b in zip(snapshots, snapshots[1:])
    ]

    print("Differences:", differences)

    assert all(diff == 10_000_000 for diff in differences)
    print("PASS")


if __name__ == "__main__":
    main()
