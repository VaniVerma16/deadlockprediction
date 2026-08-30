# Temporal Synchronization Graph Dataset v3

## Purpose

This is a synthetic temporal thread-mutex graph dataset for three-class
classification and future deadlock prediction. It was designed to remove the
main shortcuts found in v2: class-specific wait-duration ranges, trivially safe
negative examples, and evaluation on heavily overlapping validation windows.

The dataset remains fully synthetic: its lock, futex-like wait, scheduler, and
CPU-counter features are generated rather than captured by QEMU/eBPF. Results
on it must not be presented as evidence of production performance without
evaluation on independent real programs and observational telemetry.

## Size

| Split | Independent runs | Snapshots | 8-snapshot sequences |
|---|---:|---:|---:|
| Train | 1,848 | 215,051 | 51,236 |
| Validation | 396 | 45,872 | 5,570 |
| Test | 396 | 45,786 | 5,549 |
| **Total** | **2,640** | **306,709** | **62,355** |

Snapshots are spaced at 10 ms. Each sequence covers eight snapshots (70 ms
from first to last). Training uses stride 4. Validation and test use stride 8,
so their evaluation windows do not overlap.

## Scenario families

| Scenario | What it tests |
|---|---|
| `two_process_deadlock` | Basic two-thread/two-lock baseline |
| `cycle_3_5` | Longer circular dependencies involving 3-5 threads |
| `long_chain_no_cycle` | Long dependencies that never close into a cycle |
| `almost_cycle_pre_deadlock` | Near-cycles that may close or safely resolve |
| `multiple_cycles` | Simultaneous and staggered cycles |
| `cycle_with_safe_processes` | A deadlocked component mixed with unrelated safe activity |
| `resource_contention_no_deadlock` | Long, heavy contention without circular wait |
| `same_graph_different_states` | Similar current topology with outcomes determined by temporal history |
| `delayed_deadlock` | Cycles that form only after several snapshots |
| `deadlock_recovery` | Confirmed deadlocks followed by recovery and safe operation |
| `imbalanced_resource_allocation` | Highly asymmetric ownership and waiting patterns |

Each family contains randomized thread/lock counts, timings, background
contention, CPU placement, scheduler activity, and persistent versus resolving
outcomes. Validation and test are shifted toward larger graphs than training,
although their graph-size ranges partially overlap.

## Labels

Labels are assigned only after the complete run has been generated.

- `safe`: no persistent cycle is confirmed in the next 300 ms.
- `pre_deadlock`: a persistent cycle will be confirmed within the next 300 ms,
  but the current state is not yet confirmed deadlocked.
- `deadlocked`: a cycle has persisted for at least 50 ms.

This definition deliberately permits:

- long waits that remain safe;
- short-lived cycles that resolve and remain safe;
- identical or similar current topologies with different future outcomes;
- safe recovery after a previously confirmed deadlock.

Every sequence also contains binary targets for deadlock confirmation within
50 ms, 100 ms, and 300 ms. These can be used for a prediction-only experiment.

## Files

- `train.jsonl`, `validation.jsonl`, `test.jsonl`: graph snapshots.
- `*_sequences.jsonl`: references to eight consecutive snapshots and targets.
- `run_manifest.jsonl`: one record per independent execution.
- `metadata.json`: sizes, label counts, class weights, and generation settings.
- `audit_report.json`: integrity and shortcut-baseline results.
- `trainability_report.json`: small deterministic GraphSAGE-GRU pipeline check.
- `GENERATOR.md`: reproduction and customization instructions.

The Git release stores generated JSONL files as individually compressed
`*.jsonl.gz` artifacts. Use `gzip -dk dataset/v3/*.jsonl.gz` after cloning to
restore the filenames listed above. Raw JSONL files remain in the local working
tree and are reproducible with the generator.

## Model inputs and exclusions

Node and edge input fields retain the v2 schema and are compatible with the
repository's temporal data contract. Only the following should be used as inputs:

- thread features: `is_waiting`, `wait_ns`, `scheduler_switches`, `wakeups`,
  `cpu_migrations`, `last_cpu`;
- lock feature: `has_owner`;
- typed edges: `owned_by`, `waits_for`.

The scheduler, wakeup, and migration values are per-10-ms interval values, not
cumulative run-age counters.

Never use these fields as model inputs:

`label`, `label_metadata`, `has_cycle`, `cycle_nodes`, `targets`, `provenance`,
`run_id`, `snapshot_id`, or `graph_features`.

## Evaluation protocol

1. Tune only on training and validation.
2. Keep the test split untouched until model and hyperparameters are fixed.
3. Report macro-F1 and per-class precision/recall/F1, not accuracy alone.
4. For prediction, report pre-deadlock recall, false alarms per run, and median
   warning lead time.
5. Compare against the shortcut baselines recorded in `audit_report.json`.

The class distribution reflects normal operation: training sequences are about
63% safe, 17% pre-deadlock, and 20% deadlocked. Use the class weights in
`metadata.json`, balanced sampling, and macro-F1/per-class recall rather than
accuracy alone.

The generated validation set contains adjacency-aware graph topologies that
occur with more than one class label. Therefore, topology alone is not always
sufficient; temporal evolution is required for the intended task. The exact
count for this release is recorded in `metadata.json`.
