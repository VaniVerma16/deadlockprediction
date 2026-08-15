#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import subprocess
from pathlib import Path


def run(command: list[str]) -> None:
    print("+", " ".join(command))
    subprocess.run(command, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run one dataset experiment in QEMU")
    parser.add_argument("scenario", choices=["abba", "nway", "dining"])
    parser.add_argument("--mode", choices=["safe", "deadlock"], required=True)
    parser.add_argument("--threads", type=int)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--timeout-ms", type=int, default=1500)
    parser.add_argument("--iterations", type=int, default=32)
    parser.add_argument("--generation", default="manual")
    parser.add_argument("--noise", choices=["none", "cpu", "memory", "mixed"], default="none")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2222)
    parser.add_argument("--user", default="codex")
    args = parser.parse_args()
    if args.threads is None:
        args.threads = {"abba": 2, "nway": 4, "dining": 5}[args.scenario]
    if args.scenario == "abba" and args.threads != 2:
        parser.error("ABBA requires exactly 2 threads")
    if not 2 <= args.threads <= 64:
        parser.error("threads must be between 2 and 64")
    if not 1 <= args.iterations <= 10000:
        parser.error("iterations must be between 1 and 10000")

    project = Path(__file__).resolve().parents[2]
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    generation = "".join(character if character.isalnum() else "-" for character in args.generation)
    run_id = (
        f"{args.scenario}-{args.mode}-{args.noise}-t{args.threads}-s{args.seed}"
        f"-g{generation}-{stamp}"
    )
    destination = f"{args.user}@{args.host}"
    ssh = ["ssh", "-p", str(args.port), "-o", "StrictHostKeyChecking=accept-new"]

    run(["rsync", "-az", "--delete", "--exclude", "vm", "--exclude", "runs",
         "--exclude", "build", "--exclude", "bin", "--exclude", ".venv",
         "--exclude", "dataset", "--exclude", "output", "--exclude", "tmp",
         "-e", " ".join(ssh),
         f"{project}/", f"{destination}:/home/{args.user}/deadlock-dataset/"])
    remote = (
        f"cd /home/{args.user}/deadlock-dataset && "
        f"bash scripts/run_guest.sh {run_id} {args.scenario} {args.mode} "
        f"{args.threads} {args.seed} {args.timeout_ms} {args.noise} {args.iterations} "
        f"{generation}"
    )
    run([*ssh, destination, remote])
    local_run = project / "runs" / run_id
    local_run.parent.mkdir(parents=True, exist_ok=True)
    run(["rsync", "-az", "-e", " ".join(ssh),
         f"{destination}:/home/{args.user}/deadlock-dataset/runs/{run_id}/",
         f"{local_run}/"])
    print(f"collected {local_run}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
