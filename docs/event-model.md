# Event and graph model

All clocks use `CLOCK_MONOTONIC`/`bpf_ktime_get_ns`, so workload and eBPF events
can be ordered without wall-clock synchronization. One JSON object is written per
line. The required event fields are `run_id`, `ts_ns`, `source`, and `event`.

The graph uses two node types:

- `thread:<pid>:<tid>`
- `lock:<pid>:<virtual-address>`

Successful `pthread_mutex_lock` return probes create `owned_by` edges from a lock
to a thread. Futex waits create `waits_for` edges from a thread to a lock. A
strongly connected component containing more than one node is treated as a
cycle.

Labels are temporal. Snapshots before the configured lead-up window are `safe`.
Snapshots in the lead-up window and during cycle confirmation are `unsafe`. A
cycle that survives the confirmation period is `deadlocked`.

Workload JSON is ground truth and eBPF JSON is the observation stream. They are
kept separately so sensor recall can be measured before training.

The formal raw-event schema is [`../schemas/event.schema.json`](../schemas/event.schema.json).
The processed graph-snapshot schema is
[`../schemas/snapshot.schema.json`](../schemas/snapshot.schema.json). Dataset
composition, statistics, split methodology, and limitations are documented in
the repository [`DATASET_CARD.md`](../DATASET_CARD.md).
