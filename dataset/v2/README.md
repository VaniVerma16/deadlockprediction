# Temporal Synchronization Graph Dataset v2

## Summary

This dataset contains fixed-cadence temporal thread-mutex graphs reconstructed
from controlled pthread workloads in an ARM64 Ubuntu QEMU guest. It supports
three-state temporal graph classification: `safe`, `pre_deadlock`, and
`deadlocked`.

| Property | Value |
|---|---:|
| Independent QEMU runs | 720 |
| Graph snapshots | 57,651 |
| Eight-snapshot sequences | 26,486 |
| Snapshot interval | 10 ms |
| Sequence length / stride | 8 / 2 snapshots |
| Processed size | 149 MiB |
| Guest kernel | Ubuntu 6.8.0-136-generic, ARM64 |

The matrix covers ABBA, N-way circular wait, and dining-philosophers workloads;
safe and deadlocking modes; 2, 4, 5, and 8 threads where applicable; ten seeds;
and none, CPU, memory, and mixed background noise. ABBA uses two threads.

## Splits

Splits are globally seed-disjoint. A seed never appears in more than one split,
even across different scenarios, thread counts, or noise modes.

| Split | Seeds | Runs | Snapshots | Sequences |
|---|---|---:|---:|---:|
| Train | 101-107 | 504 | 40,425 | 18,561 |
| Validation | 108 | 72 | 5,726 | 2,630 |
| Test | 109-110 | 144 | 11,500 | 5,295 |

Training sequence labels are 10,903 safe, 4,382 pre-deadlock, and 3,276
deadlocked. Recommended inverse-frequency weights are stored in `metadata.json`.

## Telemetry and graph semantics

The eBPF collector combines pthread mutex uprobes/uretprobes, futex syscall
tracepoints, scheduler switch/wakeup events, and thread exit. This captures
uncontended user-space mutex operations as well as kernel-visible blocking.

- `lock -> thread` (`owned_by`) means a successful mutex acquisition.
- `thread -> lock` (`waits_for`) means the thread is blocked on that mutex.
- Thread features include waiting state/duration, scheduler switches, wakeups,
  CPU migrations, and last observed CPU.
- Lock features include current ownership state.

Each sequence references eight consecutive snapshots from one run. Snapshot
graphs are stored once in `<split>.jsonl`; compact sequence references are in
`<split>_sequences.jsonl`.

## Labels

- `safe`: no observed cycle and outside the 300 ms pre-cycle warning window.
- `pre_deadlock`: a developing state within 300 ms before the first cycle or
  during its 50 ms confirmation interval.
- `deadlocked`: the observed circular wait persists for at least 50 ms.

Deadlock runs are retained for 300 ms after the first cycle. The builder rejects
a deadlock run unless all three classes occur and at least one pre-deadlock
snapshot is strictly before the cycle.

## Validation

`validation_report.json` records a clean validation result with no errors or
warnings. Checks include all 720 matrix configurations, graph edge/node
invariants, finite numeric features, fixed 10 ms timing, consecutive sequence
references, no cross-run sequences, class semantics, required sensor coverage,
true pre-cycle examples, and run/seed split disjointness.

The relational GraphSAGE-GRU smoke test in `trainability_report.json` trained on
3,000 balanced sequences. Loss decreased from 0.213 to 0.072 with finite
gradients. On 1,200 balanced validation sequences it reached 98.75% accuracy
and macro-F1, compared with a 33.33% majority baseline. This demonstrates file
and model-path trainability; it is not a production-performance claim.

## Leakage controls and limitations

Labels, cycle flags/nodes, label metadata, provenance, identifiers, and
run-level performance aggregates are excluded from smoke-model inputs. The test
split must remain untouched until final model selection is complete.

The dataset is synthetic and controlled. It covers pthread mutex/futex behavior,
not arbitrary production software, other synchronization primitives,
distributed deadlocks, or kernel-internal lock deadlocks. Strong results should
be followed by evaluation on independent real applications and physical Linux
hosts.

## Reproduction

```bash
python3 scripts/qemu/run_matrix.py --config config/experiments.v2.json --resume
PYTHONPATH=src python3 -m deadlock_dataset.cli temporal-build \
  --runs-dir runs --output dataset/v2 --config config/experiments.v2.json
PYTHONPATH=src python3 -m deadlock_dataset.cli temporal-validate \
  --dataset dataset/v2
.venv/bin/python scripts/train_temporal_smoke.py \
  --dataset dataset/v2 --output dataset/v2/trainability_report.json
```
