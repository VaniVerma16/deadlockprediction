# Temporal Synchronization Graph Dataset

## Current release: v2

The current dataset release is [`dataset/v2`](dataset/v2). It contains
fixed-cadence temporal thread-mutex graphs reconstructed from eBPF telemetry
collected during controlled pthread workloads in an ARM64 Ubuntu QEMU guest.

| Property | v2 value |
|---|---:|
| Independent QEMU runs | **720** |
| Graph snapshots | **57,651** |
| Eight-snapshot sequences | **26,486** |
| Snapshot interval | 10 ms |
| Classes | `safe`, `pre_deadlock`, `deadlocked` |
| Processed size | 149 MiB |

The matrix covers ABBA, N-way circular wait, and dining-philosophers workloads;
safe and deadlocking modes; 2, 4, 5, and 8 threads where applicable; ten seeds;
and none, CPU, memory, and mixed background noise.

## Split composition

Splits are globally seed-disjoint: a seed never appears in more than one split,
including across different scenarios, thread counts, modes, or noise settings.

| Split | Seeds | Runs | Snapshots | Sequences |
|---|---|---:|---:|---:|
| Train | 101-107 | 504 | 40,425 | 18,561 |
| Validation | 108 | 72 | 5,726 | 2,630 |
| Test | 109-110 | 144 | 11,500 | 5,295 |
| **Total** | 10 seeds | **720** | **57,651** | **26,486** |

## Validation and trainability

The independent v2 validator reported zero errors and zero warnings. It checks
matrix completeness, graph invariants, finite features, fixed 10 ms timing,
sequence continuity, class semantics, required telemetry, genuine pre-cycle
warning states, and run/seed split leakage.

A relational GraphSAGE-GRU smoke test trained on 3,000 balanced sequences. Loss
decreased from 0.213 to 0.072 with finite gradients. On 1,200 balanced,
seed-held validation sequences, it reached 98.75% accuracy and macro-F1 against
a 33.33% majority baseline. This verifies trainability of the data pipeline; it
is not a production-performance claim.

Detailed v2 documentation and reports:

- [`dataset/v2/README.md`](dataset/v2/README.md)
- [`dataset/v2/metadata.json`](dataset/v2/metadata.json)
- [`dataset/v2/validation_report.json`](dataset/v2/validation_report.json)
- [`dataset/v2/trainability_report.json`](dataset/v2/trainability_report.json)
- [`output/pdf/temporal-synchronization-graph-dataset-v2-documentation.pdf`](output/pdf/temporal-synchronization-graph-dataset-v2-documentation.pdf)

## Historical releases

The 360-run dataset was not overwritten. It remains available as the historical
snapshot releases:

- `dataset/v0`: 360 runs and 27,329 snapshots.
- `dataset/v1`: 360 runs and 6,418 active-allocation snapshots.

The full historical v1 card is now stored at
[`dataset/v1/DATASET_CARD.md`](dataset/v1/DATASET_CARD.md).

## Reproduction

```bash
python3 scripts/qemu/run_matrix.py --config config/experiments.v2.json --resume
PYTHONPATH=src python3 -m deadlock_dataset.cli temporal-build \
  --runs-dir runs --output dataset/v2 --config config/experiments.v2.json
PYTHONPATH=src python3 -m deadlock_dataset.cli temporal-validate \
  --dataset dataset/v2
```
