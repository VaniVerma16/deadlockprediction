# QEMU/eBPF deadlock dataset

This repository generates labeled resource-allocation graph snapshots from
controlled pthread deadlock scenarios running in an ARM64 Linux QEMU guest.

For dataset composition, schema, collection methodology, quality statistics,
limitations, and loading examples, see the [dataset card](DATASET_CARD.md).

## What is implemented

- Deterministic ABBA, N-way, and dining-philosophers C workloads, each with safe
  and deadlocking modes.
- Ground-truth JSONL markers from the workloads.
- A CO-RE/libbpf collector for pthread uprobes, futex tracepoints, scheduler
  tracepoints, and process exit.
- `perf stat` collection for software and available hardware counters.
- Resource-allocation graph construction, Tarjan cycle detection, temporal
  `safe`/`unsafe`/`deadlocked` labels, and dataset validation.
- ARM64 Ubuntu cloud-init and QEMU/HVF scripts for Apple Silicon.
- Single-run and experiment-matrix host controllers.

The raw event contract is documented in [docs/event-model.md](docs/event-model.md).
The processed snapshot schema is [schemas/snapshot.schema.json](schemas/snapshot.schema.json).
The curated run manifest schema is [schemas/run-manifest.schema.json](schemas/run-manifest.schema.json).

## Local verification

The workloads and graph builder can run on macOS without QEMU:

```bash
make workloads
./bin/abba --mode deadlock --run-id local-abba --timeout-ms 300 --start-delay-ms 0
make test
make demo
```

macOS cannot load the Linux eBPF collector. That part is built and run in the
guest.

## Create the Linux guest

Install QEMU and make sure an Ed25519 public key exists:

```bash
brew install qemu
scripts/qemu/create_vm.sh ~/.ssh/id_ed25519.pub
scripts/qemu/start_vm.sh
```

The first boot installs the compiler, libbpf, bpftool, perf, Python, and rsync.
Wait for the cloud-init completion message. SSH is forwarded to port 2222:

```bash
ssh -p 2222 codex@127.0.0.1
```

## Generate one run

With the VM running in another terminal:

```bash
python3 scripts/qemu/run_experiment.py abba --mode deadlock --threads 2 --seed 1 --noise cpu
```

The host receives:

```text
runs/<run-id>/
  workload.jsonl
  ebpf.jsonl
  perf.csv
  snapshots.jsonl
  run_summary.json
  collector.stderr
```

Run a small matrix after inspecting the first few runs:

```bash
python3 scripts/qemu/run_matrix.py --config config/experiments.example.json
```

Generate the sequence-ready temporal `v2` matrix and resume safely after an
interruption:

```bash
python3 scripts/qemu/run_matrix.py --config config/experiments.v2.json --resume
PYTHONPATH=src python3 -m deadlock_dataset.cli temporal-build \
  --runs-dir runs --output dataset/v2 --config config/experiments.v2.json
PYTHONPATH=src python3 -m deadlock_dataset.cli temporal-validate \
  --dataset dataset/v2
```

The temporal build uses fixed 10 ms graph snapshots, 8-snapshot sequences,
`safe` / `pre_deadlock` / `deadlocked` targets, and globally seed-disjoint
train/validation/test splits. It excludes labels, cycle results, identifiers,
and run-level perf aggregates from the model feature tensors. A small relational
GraphSAGE-GRU trainability check can be run with:

```bash
.venv/bin/python scripts/train_temporal_smoke.py \
  --dataset dataset/v2 --output dataset/v2/trainability_report.json
```

Rebuild active-allocation snapshots after the matrix completes:

```bash
PYTHONPATH=src python3 -m deadlock_dataset.cli rebuild \
  --runs-dir runs --snapshot-name snapshots-v1.jsonl \
  --interval-ms 20 --unsafe-window-ms 250 \
  --confirm-ms 50 --post-cycle-ms 250
```

Create leakage-free `v1` splits. Entire runs stay together:

```bash
PYTHONPATH=src python3 -m deadlock_dataset.cli split \
  --runs-dir runs --snapshot-name snapshots-v1.jsonl \
  --output dataset/v1 --seed 20260725
```

## Dataset semantics

Successful lock returns create `lock -> thread` ownership edges. Futex waits
create `thread -> lock` waiting edges. Persistent graph cycles are deadlocked;
the configured lead-up and confirmation windows are unsafe. Perf values are
stored as run-level graph features in v1. A later collector revision can sample
them per time window.

The workload stream is ground truth and the eBPF stream is observed data. Keep
both: their differences measure sensor recall and expose libc/kernel-specific
blind spots before GNN training.
