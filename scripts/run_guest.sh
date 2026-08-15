#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 9 ]]; then
  echo "usage: $0 RUN_ID SCENARIO MODE THREADS SEED TIMEOUT_MS NOISE ITERATIONS GENERATION_ID" >&2
  exit 2
fi

RUN_ID=$1
SCENARIO=$2
MODE=$3
THREADS=$4
SEED=$5
TIMEOUT_MS=$6
NOISE=$7
ITERATIONS=$8
GENERATION_ID=$9
PROJECT_DIR=$(cd "$(dirname "$0")/.." && pwd)
RUN_DIR="$PROJECT_DIR/runs/$RUN_ID"
WORKLOAD="$PROJECT_DIR/bin/$SCENARIO"
COLLECTOR="$PROJECT_DIR/build/collector/deadlock-collector"

mkdir -p "$RUN_DIR"
cd "$PROJECT_DIR"
make workloads
make -C collector ARCH="$(uname -m)"

NOISE_PID=""
if [[ "$NOISE" != "none" ]]; then
  case "$NOISE" in
    cpu) stress-ng --cpu 1 --timeout "$((TIMEOUT_MS + 2000))ms" --quiet & ;;
    memory) stress-ng --vm 1 --vm-bytes 128M --timeout "$((TIMEOUT_MS + 2000))ms" --quiet & ;;
    mixed) stress-ng --cpu 1 --vm 1 --vm-bytes 128M --timeout "$((TIMEOUT_MS + 2000))ms" --quiet & ;;
    *) echo "noise must be none, cpu, memory, or mixed" >&2; exit 2 ;;
  esac
  NOISE_PID=$!
fi

"$WORKLOAD" --run-id "$RUN_ID" --mode "$MODE" --threads "$THREADS" \
  --seed "$SEED" --timeout-ms "$TIMEOUT_MS" --start-delay-ms 750 \
  --iterations "$ITERATIONS" \
  >"$RUN_DIR/workload.jsonl" 2>"$RUN_DIR/workload.stderr" &
WORKLOAD_PID=$!

LIBC_PATH=$(ldd "$WORKLOAD" | awk '/libc\.so/ {print $3; exit}')
if [[ -z "$LIBC_PATH" ]]; then
  echo "could not locate libc for $WORKLOAD" >&2
  kill "$WORKLOAD_PID" || true
  exit 1
fi

sudo "$COLLECTOR" --pid "$WORKLOAD_PID" --run-id "$RUN_ID" \
  --libc "$LIBC_PATH" --output "$RUN_DIR/ebpf.jsonl" \
  >"$RUN_DIR/collector.stdout" 2>"$RUN_DIR/collector.stderr" &
COLLECTOR_PID=$!

sudo sysctl -w kernel.perf_event_paranoid=1 >/dev/null
perf stat -x, -o "$RUN_DIR/perf.csv" \
  -e task-clock,context-switches,cpu-migrations,page-faults,cycles,instructions,cache-misses \
  -p "$WORKLOAD_PID" -- sleep "$(awk "BEGIN {print ($TIMEOUT_MS + 1500) / 1000}")" &
PERF_PID=$!

set +e
wait "$WORKLOAD_PID"
WORKLOAD_STATUS=$?
set -e
sudo kill -TERM "$COLLECTOR_PID" 2>/dev/null || true
wait "$COLLECTOR_PID" 2>/dev/null || true
kill -INT "$PERF_PID" 2>/dev/null || true
wait "$PERF_PID" 2>/dev/null || true
if [[ -n "$NOISE_PID" ]]; then
  kill "$NOISE_PID" 2>/dev/null || true
  wait "$NOISE_PID" 2>/dev/null || true
fi

PYTHONPATH=src python3 -m deadlock_dataset.cli build \
  --events "$RUN_DIR/workload.jsonl" "$RUN_DIR/ebpf.jsonl" \
  --perf "$RUN_DIR/perf.csv" --output "$RUN_DIR/snapshots.jsonl"
PYTHONPATH=src python3 -m deadlock_dataset.cli validate \
  --dataset "$RUN_DIR/snapshots.jsonl" --expected-mode "$MODE"

python3 - "$RUN_DIR/run_summary.json" <<PY
import json, platform, sys
json.dump({
    "run_id": "$RUN_ID",
    "scenario": "$SCENARIO",
    "mode": "$MODE",
    "threads": int("$THREADS"),
    "seed": int("$SEED"),
    "timeout_ms": int("$TIMEOUT_MS"),
    "noise": "$NOISE",
    "iterations": int("$ITERATIONS"),
    "generation_id": "$GENERATION_ID",
    "workload_status": int("$WORKLOAD_STATUS"),
    "kernel": platform.release(),
    "machine": platform.machine(),
}, open(sys.argv[1], "w"), indent=2)
PY

echo "$RUN_DIR"
