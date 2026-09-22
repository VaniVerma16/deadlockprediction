#!/usr/bin/env python3
"""
End-to-end live synchronization Digital Twin + V5 monitor.

Run on Linux. The monitored application should already be running.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(ROOT))

from deadlock_prototype.digital_twin import SynchronizationDigitalTwin
from deadlock_prototype.events import RuntimeEvent
from deadlockprediction.kernel_deadlock_prototype.experiments.ebpf_collector import EBPFCollector


def snapshot_to_dict(snapshot):
    if hasattr(snapshot, "model_dump"):
        return snapshot.model_dump()
    return snapshot.dict()


def print_prediction(prediction: dict, snapshot: dict) -> None:
    print("\n" + "=" * 70)
    print("PREDICTIVE DEADLOCK MONITOR")
    print("=" * 70)

    print(f"Snapshot: {snapshot['snapshot_id']}")
    print(f"Time: {snapshot['ts_ns']}")
    print(f"State: {prediction['state']}")

    print(
        f"P(deadlock): "
        f"{prediction['p_deadlock']:.4f}"
    )
    print(
        f"P(pre-deadlock): "
        f"{prediction['p_pre']:.4f}"
    )
    print(
        f"Risk <=50ms: "
        f"{prediction['risk_50']:.4f}"
    )
    print(
        f"Risk <=100ms: "
        f"{prediction['risk_100']:.4f}"
    )
    print(
        f"Risk <=300ms: "
        f"{prediction['risk_300']:.4f}"
    )

    if prediction["state"] == "pre_deadlock":
        print("\nEARLY WARNING: synchronization state is approaching deadlock.")

    elif prediction["state"] == "deadlocked":
        print("\nDEADLOCK WARNING: deadlock state predicted/detected.")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pid", type=int, required=True)
    args = parser.parse_args()

    # Import after ROOT has been added so the existing V5 implementation
    # can be reused without modifying the trained checkpoint.
    from deadlockprediction.kernel_deadlock_prototype.v5_inference import V5Inference

    collector = EBPFCollector(pid=args.pid)
    twin = SynchronizationDigitalTwin()
    model = V5Inference()

    print("Live predictive deadlock monitor started.")
    print(f"Application PID: {args.pid}")
    print("Model: existing V5Inference default checkpoint")
    print("Sampling interval: 10 ms")
    print("Press Ctrl+C to stop.")

    next_event_index = 0
    snapshot_counter = 0

    try:
        while True:
            collector.poll(timeout_ms=50)

            # Consume newly collected normalized events.
            while next_event_index < len(collector.events):
                event = collector.events[next_event_index]
                next_event_index += 1
                twin.update(event)

            if not collector.events:
                continue

            now_ns = collector.events[-1].ts_ns

            while twin.ready(now_ns):
                snapshot = twin.sample(
                    f"live-{snapshot_counter:06d}"
                )
                snapshot_counter += 1

                if snapshot is None:
                    break

                snapshot_dict = snapshot_to_dict(snapshot)

                # Existing V5Inference accepts the graph snapshot directly.
                prediction = model.add_snapshot(snapshot_dict)

                if prediction is not None:
                    print_prediction(
                        prediction,
                        snapshot_dict,
                    )

    except KeyboardInterrupt:
        print("\nStopping monitor.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
