#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Execute an experiment matrix serially")
    parser.add_argument("--config", type=Path, default=Path("config/experiments.example.json"))
    parser.add_argument("--port", type=int, default=2222)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    runner = Path(__file__).with_name("run_experiment.py")
    generation = str(config.get("generation_id", "manual"))
    completed: set[tuple[str, str, int, int, str]] = set()
    if args.resume:
        project = Path(__file__).resolve().parents[2]
        for summary_path in (project / "runs").glob("*/run_summary.json"):
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            run_dir = summary_path.parent
            if summary.get("generation_id") != generation:
                continue
            if not all((run_dir / name).exists() for name in (
                "workload.jsonl", "ebpf.jsonl", "snapshots.jsonl"
            )):
                continue
            completed.add((
                str(summary["scenario"]), str(summary["mode"]), int(summary["threads"]),
                int(summary["seed"]), str(summary["noise"]),
            ))

    for scenario in config["scenarios"]:
        for mode in config["modes"]:
            for threads in config["thread_counts"]:
                if scenario == "abba" and threads != 2:
                    continue
                for seed in config["seeds"]:
                    for noise in config.get("noise", ["none"]):
                        key = (scenario, mode, threads, seed, noise)
                        if key in completed:
                            print("skip completed", key, flush=True)
                            continue
                        command = [
                            sys.executable, str(runner), scenario,
                            "--mode", mode,
                            "--threads", str(threads),
                            "--seed", str(seed),
                            "--timeout-ms", str(config["timeout_ms"]),
                            "--iterations", str(config.get("iterations", 32)),
                            "--generation", generation,
                            "--noise", noise,
                            "--host", args.host,
                            "--port", str(args.port),
                        ]
                        print("+", " ".join(command), flush=True)
                        subprocess.run(command, check=True)
                        completed.add(key)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
