#!/usr/bin/env python3
"""
Live WebSocket backend for real-time predictive deadlock detection.

This module is intentionally thin. It reuses, unmodified:
  - experiments/ebpf_collector.py        (EBPFCollector)
  - src/deadlock_prototype/digital_twin.py (SynchronizationDigitalTwin)
  - v5_inference.py                      (V5Inference)

It adds only the piece that was missing: a loop that drives those three
components in real time and a WebSocket layer that broadcasts the
combined graph + inference JSON contract to any connected frontend.

Design notes:
  - The monitoring/model loop (eBPF -> Digital Twin -> V5) runs on a
    dedicated background thread. It never awaits on network I/O.
  - Messages are handed to the asyncio event loop via
    loop.call_soon_threadsafe(...), so a slow or disconnected frontend
    can never backpressure the 10 ms sampling loop.
  - The Digital Twin owns the 10 ms virtual clock (see
    test_digital_twin.py: consecutive twin.sample() calls are exactly
    10_000_000 ns apart). This loop just calls twin.sample() at a
    matching real-time cadence so live snapshots stay paced to reality.

Run (same privilege pattern as the existing live_monitor.py):

    sudo env PYTHONPATH=/usr/lib/python3/dist-packages \
        $(pwd)/.venv/bin/python3 server/live_server.py --pid <PID>
"""

from __future__ import annotations

import argparse
import asyncio
import json
import queue
import sys
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
EXPERIMENTS = ROOT / "experiments"

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(EXPERIMENTS))

import websockets  # pip install websockets

from deadlock_prototype.digital_twin import SynchronizationDigitalTwin  # noqa: E402
from ebpf_collector import EBPFCollector  # noqa: E402
from v5_inference import V5Inference  # noqa: E402


SAMPLE_INTERVAL_S = 0.010  # 10 ms — matches the Digital Twin's verified interval
EVENT_LOG_MAXLEN = 50      # "latest ~20-50 synchronization events" per spec
EBPF_POLL_TIMEOUT_MS = 5   # short poll so we never overshoot the 10 ms cadence


# ---------------------------------------------------------------------
# Snapshot / event serialization helpers
# ---------------------------------------------------------------------

def snapshot_to_payload(snapshot: Any) -> dict[str, Any]:
    """
    Normalize whatever the Digital Twin returns from twin.sample(...)
    into the plain-dict graph contract used both by v5_inference and
    the WebSocket message (see section 9 of the handoff doc).

    Handles pydantic v1 (.dict()), pydantic v2 (.model_dump()), and
    plain-object / dict snapshots defensively, without assuming which
    one digital_twin.py actually returns.
    """
    if isinstance(snapshot, dict):
        data = snapshot
    elif hasattr(snapshot, "model_dump"):
        data = snapshot.model_dump()
    elif hasattr(snapshot, "dict"):
        data = snapshot.dict()
    else:
        # Fall back to reading the attributes the rest of the pipeline
        # (v5_inference.tensorize_graph, convert_events.py) relies on.
        data = {
            "snapshot_id": getattr(snapshot, "snapshot_id", None),
            "ts_ns": getattr(snapshot, "ts_ns", None),
            "nodes": getattr(snapshot, "nodes", []),
            "edges": getattr(snapshot, "edges", []),
        }

    return {
        "snapshot_id": data.get("snapshot_id"),
        "ts_ns": data.get("ts_ns"),
        "nodes": data.get("nodes", []),
        "edges": data.get("edges", []),
    }


def event_to_log_entry(event: Any) -> dict[str, Any]:
    """Compact representation of a RuntimeEvent for the frontend Event Log."""
    return {
        "ts_ns": getattr(event, "ts_ns", None),
        "event": getattr(event, "event", None),
        "tid": getattr(event, "tid", None),
        "lock_id": getattr(event, "lock_id", None),
        "owner_tid": getattr(event, "owner_tid", None),
        "cpu": getattr(event, "cpu", None),
    }


# ---------------------------------------------------------------------
# Monitor loop (runs on a background thread)
# ---------------------------------------------------------------------

class MonitorLoop:
    """
    Drives EBPFCollector -> SynchronizationDigitalTwin -> V5Inference and
    hands finished messages to the asyncio side via a thread-safe queue.
    """

    def __init__(
        self,
        pid: int,
        loop: asyncio.AbstractEventLoop,
        out_queue: "asyncio.Queue[str]",
        model_path: Path | None = None,
    ):
        self.pid = pid
        self.loop = loop
        self.out_queue = out_queue
        self.model_path = model_path

        self._stop = threading.Event()
        self._snapshot_counter = 0
        self._recent_events: deque = deque(maxlen=EVENT_LOG_MAXLEN)

        print(f"[monitor] attaching eBPF collector to PID={self.pid}")
        self.collector = EBPFCollector(pid=self.pid)

        print("[monitor] initializing Synchronization Digital Twin")
        self.twin = SynchronizationDigitalTwin()

        print("[monitor] loading V5 model (this reuses the trained checkpoint)")
        if self.model_path is not None:
            self.model = V5Inference(model_path=self.model_path)
        else:
            self.model = V5Inference()

    def stop(self) -> None:
        self._stop.set()

    def _drain_new_events(self) -> list:
        """Pull any events the collector's perf-buffer callback has queued
        since the last drain, feed them to the Digital Twin, and return
        them for the event log."""
        new_events = self.collector.events
        self.collector.events = []

        for event in new_events:
            self.twin.update(event)
            self._recent_events.append(event_to_log_entry(event))

        return new_events

    def _publish(self, message: dict[str, Any]) -> None:
        payload = json.dumps(message)
        # Thread-safe hand-off into the asyncio world.
        self.loop.call_soon_threadsafe(self.out_queue.put_nowait, payload)

    def run(self) -> None:
        print(f"[monitor] sampling every {SAMPLE_INTERVAL_S * 1000:.0f} ms")
        next_sample_at = time.monotonic()

        try:
            while not self._stop.is_set():
                # Drain whatever eBPF events have arrived without
                # blocking past our sampling cadence.
                self.collector.poll(timeout_ms=EBPF_POLL_TIMEOUT_MS)
                self._drain_new_events()

                now = time.monotonic()
                if now < next_sample_at:
                    continue
                next_sample_at += SAMPLE_INTERVAL_S
                if next_sample_at < now:
                    # We fell behind (e.g. slow eBPF poll) — resync
                    # rather than firing a burst of catch-up samples.
                    next_sample_at = now + SAMPLE_INTERVAL_S

                self._snapshot_counter += 1
                snapshot_id = f"live-{self._snapshot_counter:06d}"

                snapshot = self.twin.sample(snapshot_id)
                graph_payload = snapshot_to_payload(snapshot)

                # V5Inference.add_snapshot expects the plain-dict graph
                # contract (nodes/edges) — exactly what tensorize_graph
                # in v5_inference.py consumes.
                snapshot_dict = {
                    "nodes": graph_payload["nodes"],
                    "edges": graph_payload["edges"],
                }
                result = self.model.add_snapshot(snapshot_dict)

                if not result.get("ready", False):
                    # Still warming up (< 8 snapshots). Per spec, do not
                    # invent predictions during warm-up.
                    message = {
                        "snapshot_id": graph_payload["snapshot_id"],
                        "ts_ns": graph_payload["ts_ns"],
                        "graph": {
                            "nodes": graph_payload["nodes"],
                            "edges": graph_payload["edges"],
                        },
                        "inference": {"ready": False, **result},
                        "events": list(self._recent_events),
                    }
                else:
                    message = {
                        "snapshot_id": graph_payload["snapshot_id"],
                        "ts_ns": graph_payload["ts_ns"],
                        "graph": {
                            "nodes": graph_payload["nodes"],
                            "edges": graph_payload["edges"],
                        },
                        "inference": {
                            "ready": True,
                            "state": result["state"],
                            "deadlock_probability": result["deadlock_probability"],
                            "pre_deadlock_probability": result["pre_deadlock_probability"],
                            "risk_50ms": result["risk_50ms"],
                            "risk_100ms": result["risk_100ms"],
                            "risk_300ms": result["risk_300ms"],
                            "thresholds": result["thresholds"],
                        },
                        "events": list(self._recent_events),
                    }

                self._publish(message)

        except KeyboardInterrupt:
            pass
        finally:
            print("[monitor] stopped")


# ---------------------------------------------------------------------
# WebSocket layer
# ---------------------------------------------------------------------

CONNECTIONS: set = set()


async def handler(websocket) -> None:
    CONNECTIONS.add(websocket)
    peer = getattr(websocket, "remote_address", "?")
    print(f"[ws] client connected: {peer}")
    try:
        # This backend is broadcast-only; we don't expect inbound
        # messages, but we still need to await the connection so the
        # handler stays alive and cleans up on disconnect.
        await websocket.wait_closed()
    finally:
        CONNECTIONS.discard(websocket)
        print(f"[ws] client disconnected: {peer}")


async def broadcaster(out_queue: "asyncio.Queue[str]") -> None:
    while True:
        message = await out_queue.get()
        if not CONNECTIONS:
            continue
        # Send concurrently; drop clients that error out.
        results = await asyncio.gather(
            *(ws.send(message) for ws in list(CONNECTIONS)),
            return_exceptions=True,
        )
        for ws, result in zip(list(CONNECTIONS), results):
            if isinstance(result, Exception):
                CONNECTIONS.discard(ws)


async def async_main(pid: int, host: str, port: int, model_path: Path | None) -> None:
    loop = asyncio.get_running_loop()
    out_queue: "asyncio.Queue[str]" = asyncio.Queue()

    monitor = MonitorLoop(pid=pid, loop=loop, out_queue=out_queue, model_path=model_path)
    monitor_thread = threading.Thread(target=monitor.run, name="monitor-loop", daemon=True)
    monitor_thread.start()

    async with websockets.serve(handler, host, port):
        print(f"[ws] listening on ws://{host}:{port}/ws")
        try:
            await broadcaster(out_queue)
        finally:
            monitor.stop()


def main() -> int:
    parser = argparse.ArgumentParser(description="Live deadlock-detection WebSocket backend")
    parser.add_argument("--pid", type=int, required=True, help="PID of the monitored application")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="Override outputs/gnn_v5/best_model_v5.pt (defaults to v5_inference.MODEL_PATH)",
    )
    args = parser.parse_args()

    try:
        asyncio.run(async_main(args.pid, args.host, args.port, args.model_path))
    except KeyboardInterrupt:
        print("\n[live_server] shutting down")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
