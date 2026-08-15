# Deadlock RAG dataset v1

This is the current active-allocation release of the deadlock
resource-allocation graph dataset.

| Split | Runs | Snapshots |
|---|---:|---:|
| Train | 216 | 3,834 |
| Validation | 72 | 1,287 |
| Test | 72 | 1,297 |

Every graph has at least one node and one directed ownership or wait edge.
Splits are run-disjoint and stratified by scenario, mode, noise, and thread
count. See [`../../DATASET_CARD.md`](../../DATASET_CARD.md) for the full schema,
collection methodology, statistics, limitations, and reproduction commands.
