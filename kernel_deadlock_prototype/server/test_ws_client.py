#!/usr/bin/env python3
"""
Minimal WebSocket client for validating live_server.py before building
the React frontend (step 4 of the implementation order).

Usage:
    python3 server/test_ws_client.py --url ws://<VM_IP>:8000/ws
"""
from __future__ import annotations

import argparse
import asyncio
import json

import websockets


async def run(url: str) -> None:
    print(f"Connecting to {url} ...")
    async with websockets.connect(url) as ws:
        print("Connected. Waiting for messages (Ctrl+C to stop)...\n")
        while True:
            raw = await ws.recv()
            message = json.loads(raw)

            inference = message.get("inference", {})
            graph = message.get("graph", {})
            events = message.get("events", [])

            if not inference.get("ready", False):
                print(
                    f"[{message.get('snapshot_id')}] warming up "
                    f"({inference.get('snapshots')}/{inference.get('required')})"
                )
                continue

            print(
                f"[{message.get('snapshot_id')}] "
                f"state={inference['state']:<12} "
                f"deadlock={inference['deadlock_probability']:.3f} "
                f"pre={inference['pre_deadlock_probability']:.3f} "
                f"risk50={inference['risk_50ms']:.3f} "
                f"risk100={inference['risk_100ms']:.3f} "
                f"risk300={inference['risk_300ms']:.3f} "
                f"nodes={len(graph.get('nodes', []))} "
                f"edges={len(graph.get('edges', []))} "
                f"recent_events={len(events)}"
            )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", type=str, default="ws://localhost:8000/ws")
    args = parser.parse_args()

    try:
        asyncio.run(run(args.url))
    except KeyboardInterrupt:
        print("\nStopped.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
