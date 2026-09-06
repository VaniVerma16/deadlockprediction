#!/usr/bin/env python3
"""
Test Temporal Hierarchical Risk-Aware GraphSAGE + GRU V5.

IMPORTANT:
- No threshold tuning is performed on the test set.
- Thresholds saved from validation are used exactly.
- Current deadlock is checked before PRE.
- PRE is only considered for non-deadlocked samples.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


LABELS = (
    "safe",
    "pre_deadlock",
    "deadlocked",
)

LABEL_TO_ID = {
    label: i
    for i, label in enumerate(LABELS)
}

RELATIONS = (
    "owned_by",
    "waits_for",
)


# ---------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------

def load_jsonl(
    path: Path,
) -> list[dict[str, Any]]:

    rows = []

    if path.suffix == ".gz":
        handle = gzip.open(
            path,
            "rt",
            encoding="utf-8",
        )
    else:
        handle = path.open(
            "r",
            encoding="utf-8",
        )

    with handle:

        for line in handle:

            line = line.strip()

            if line:
                rows.append(
                    json.loads(line)
                )

    return rows


def resolve_jsonl_path(
    dataset_dir: Path,
    filename: str,
) -> Path:

    compressed = (
        dataset_dir
        / f"{filename}.gz"
    )

    plain = (
        dataset_dir
        / filename
    )

    if compressed.exists():
        return compressed

    if plain.exists():
        return plain

    raise FileNotFoundError(
        f"Could not find either "
        f"{compressed} or {plain}"
    )


# ---------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------

def node_features(
    node: dict[str, Any],
) -> list[float]:

    node_type = node.get(
        "type",
        "",
    )

    features = node.get(
        "features",
        {},
    )

    return [
        float(
            node_type == "thread"
        ),

        float(
            node_type == "lock"
        ),

        float(
            features.get(
                "is_waiting",
                0,
            )
        ),

        math.log1p(
            float(
                features.get(
                    "wait_ns",
                    0,
                )
            )
        ) / 22.0,

        float(
            features.get(
                "has_owner",
                0,
            )
        ),

        math.log1p(
            float(
                features.get(
                    "scheduler_switches",
                    0,
                )
            )
        ) / 8.0,

        math.log1p(
            float(
                features.get(
                    "wakeups",
                    0,
                )
            )
        ) / 8.0,

        math.log1p(
            float(
                features.get(
                    "cpu_migrations",
                    0,
                )
            )
        ) / 6.0,

        float(
            features.get(
                "last_cpu",
                0,
            )
        ) / 7.0,
    ]


def tensorize_graph(
    snapshot: dict[str, Any],
) -> dict[str, Any]:

    nodes = snapshot[
        "nodes"
    ]

    node_index = {
        node["id"]: i
        for i, node in enumerate(nodes)
    }

    x = torch.tensor(
        [
            node_features(node)
            for node in nodes
        ],
        dtype=torch.float32,
    )

    edge_lists = {
        relation: []
        for relation in RELATIONS
    }

    for edge in snapshot.get(
        "edges",
        [],
    ):

        relation = edge[
            "type"
        ]

        if relation not in edge_lists:
            continue

        source = node_index[
            edge["source"]
        ]

        target = node_index[
            edge["target"]
        ]

        edge_lists[
            relation
        ].append(
            (
                source,
                target,
            )
        )

    edges = {}

    for relation in RELATIONS:

        pairs = edge_lists[
            relation
        ]

        if pairs:

            edges[relation] = (
                torch.tensor(
                    pairs,
                    dtype=torch.long,
                )
                .t()
                .contiguous()
            )

        else:

            edges[relation] = (
                torch.empty(
                    (2, 0),
                    dtype=torch.long,
                )
            )

    return {
        "x": x,
        "edges": edges,
    }


# ---------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------

class TemporalSequenceDataset:

    def __init__(
        self,
        dataset_dir: Path,
        split: str,
    ):

        snapshot_rows = load_jsonl(
            resolve_jsonl_path(
                dataset_dir,
                f"{split}.jsonl",
            )
        )

        sequence_rows = load_jsonl(
            resolve_jsonl_path(
                dataset_dir,
                f"{split}_sequences.jsonl",
            )
        )

        self.graphs = {
            row["snapshot_id"]:
                tensorize_graph(row)
            for row in snapshot_rows
        }

        self.sequences = []

        for row in sequence_rows:

            graphs = [
                self.graphs[sid]
                for sid in row[
                    "snapshot_ids"
                ]
            ]

            label = LABEL_TO_ID[
                row["label"]
            ]

            self.sequences.append(
                (
                    graphs,
                    label,
                    row,
                )
            )

        print(
            f"Loaded "
            f"{len(snapshot_rows):,} snapshots"
        )

        print(
            f"Loaded "
            f"{len(self.sequences):,} sequences"
        )

    def __len__(self):
        return len(
            self.sequences
        )

    def __getitem__(
        self,
        index,
    ):
        return self.sequences[
            index
        ]


# ---------------------------------------------------------------------
# GraphSAGE
# ---------------------------------------------------------------------

class RelationalGraphSAGE(
    nn.Module
):

    def __init__(
        self,
        input_size=9,
        hidden_size=32,
    ):

        super().__init__()

        self.input_proj = nn.Linear(
            input_size,
            hidden_size,
        )

        self.self_layer = nn.Linear(
            hidden_size,
            hidden_size,
        )

        self.relation_layers = (
            nn.ModuleDict(
                {
                    relation:
                        nn.Linear(
                            hidden_size,
                            hidden_size,
                            bias=False,
                        )
                    for relation
                    in RELATIONS
                }
            )
        )

        self.norm = nn.LayerNorm(
            hidden_size
        )

    def aggregate(
        self,
        h,
        edge_index,
        transform,
    ):

        if edge_index.numel() == 0:
            return torch.zeros_like(h)

        source = edge_index[0]
        target = edge_index[1]

        messages = transform(
            h[source]
        )

        aggregated = torch.zeros_like(
            h
        )

        aggregated.index_add_(
            0,
            target,
            messages,
        )

        counts = torch.zeros(
            h.size(0),
            device=h.device,
            dtype=h.dtype,
        )

        counts.index_add_(
            0,
            target,
            torch.ones(
                target.size(0),
                device=h.device,
                dtype=h.dtype,
            ),
        )

        counts = counts.clamp_min(
            1.0
        ).unsqueeze(1)

        return (
            aggregated
            / counts
        )

    def forward(
        self,
        graph,
    ):

        x = graph["x"]
        edges = graph["edges"]

        h = F.relu(
            self.input_proj(x)
        )

        out = self.self_layer(h)

        for relation in RELATIONS:

            out = (
                out
                + self.aggregate(
                    h,
                    edges[relation],
                    self.relation_layers[
                        relation
                    ],
                )
            )

        h = F.relu(
            self.norm(out)
        )

        thread_mask = (
            x[:, 0] > 0.5
        )

        lock_mask = (
            x[:, 1] > 0.5
        )

        if thread_mask.any():

            thread_pool = h[
                thread_mask
            ].mean(dim=0)

        else:

            thread_pool = h.mean(
                dim=0
            )

        if lock_mask.any():

            lock_pool = h[
                lock_mask
            ].mean(dim=0)

        else:

            lock_pool = h.mean(
                dim=0
            )

        return torch.cat(
            [
                thread_pool,
                lock_pool,
            ],
            dim=0,
        )


# ---------------------------------------------------------------------
# V5 model
# ---------------------------------------------------------------------

class TemporalRiskAwareV5(
    nn.Module
):

    def __init__(
        self,
        input_size=9,
        hidden_size=32,
    ):

        super().__init__()

        self.gnn = (
            RelationalGraphSAGE(
                input_size,
                hidden_size,
            )
        )

        self.gru = nn.GRU(
            input_size=hidden_size * 2,
            hidden_size=hidden_size,
            batch_first=True,
        )

        self.norm = nn.LayerNorm(
            hidden_size
        )

        self.deadlock_head = (
            nn.Linear(
                hidden_size,
                1,
            )
        )

        self.pre_head = (
            nn.Linear(
                hidden_size,
                1,
            )
        )

        self.risk_50_head = (
            nn.Linear(
                hidden_size,
                1,
            )
        )

        self.risk_100_head = (
            nn.Linear(
                hidden_size,
                1,
            )
        )

        self.risk_300_head = (
            nn.Linear(
                hidden_size,
                1,
            )
        )

    def encode(
        self,
        batch_graph_sequences,
        device,
    ):

        sequence_embeddings = []

        for graphs in (
            batch_graph_sequences
        ):

            embeddings = []

            for graph in graphs:

                graph = {
                    "x":
                        graph["x"].to(
                            device
                        ),

                    "edges": {
                        relation:
                            edge.to(
                                device
                            )
                        for relation, edge
                        in graph[
                            "edges"
                        ].items()
                    },
                }

                embeddings.append(
                    self.gnn(graph)
                )

            sequence_embeddings.append(
                torch.stack(
                    embeddings
                )
            )

        x = torch.stack(
            sequence_embeddings
        )

        output, _ = self.gru(x)

        return self.norm(
            output[:, -1, :]
        )

    def forward(
        self,
        batch_graph_sequences,
        device,
    ):

        h = self.encode(
            batch_graph_sequences,
            device,
        )

        return {
            "deadlock":
                self.deadlock_head(
                    h
                ).squeeze(1),

            "pre":
                self.pre_head(
                    h
                ).squeeze(1),

            "risk_50":
                self.risk_50_head(
                    h
                ).squeeze(1),

            "risk_100":
                self.risk_100_head(
                    h
                ).squeeze(1),

            "risk_300":
                self.risk_300_head(
                    h
                ).squeeze(1),
        }


# ---------------------------------------------------------------------
# Hierarchical decision
# ---------------------------------------------------------------------

def hierarchical_predictions(
    deadlock_probability,
    pre_probability,
    deadlock_threshold,
    pre_threshold,
):

    predictions = np.zeros(
        len(deadlock_probability),
        dtype=np.int64,
    )

    deadlocked_mask = (
        deadlock_probability
        >= deadlock_threshold
    )

    predictions[
        deadlocked_mask
    ] = LABEL_TO_ID[
        "deadlocked"
    ]

    non_deadlocked = (
        ~deadlocked_mask
    )

    pre_mask = (
        pre_probability
        >= pre_threshold
    )

    predictions[
        non_deadlocked
        & pre_mask
    ] = LABEL_TO_ID[
        "pre_deadlock"
    ]

    predictions[
        non_deadlocked
        & ~pre_mask
    ] = LABEL_TO_ID[
        "safe"
    ]

    return predictions


# ---------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------

def calculate_metrics(
    predictions,
    targets,
):

    confusion = np.zeros(
        (3, 3),
        dtype=np.int64,
    )

    for target, prediction in zip(
        targets,
        predictions,
    ):

        confusion[
            target,
            prediction,
        ] += 1

    precisions = []
    recalls = []
    f1s = []

    per_class = {}

    for class_id, label in enumerate(
        LABELS
    ):

        tp = confusion[
            class_id,
            class_id,
        ]

        fp = (
            confusion[:, class_id]
            .sum()
            - tp
        )

        fn = (
            confusion[class_id, :]
            .sum()
            - tp
        )

        precision = (
            tp
            / max(
                1,
                tp + fp,
            )
        )

        recall = (
            tp
            / max(
                1,
                tp + fn,
            )
        )

        f1 = (
            2
            * precision
            * recall
            / max(
                1e-12,
                precision + recall,
            )
        )

        precisions.append(
            precision
        )

        recalls.append(
            recall
        )

        f1s.append(
            f1
        )

        per_class[label] = {
            "precision":
                float(precision),

            "recall":
                float(recall),

            "f1":
                float(f1),

            "support":
                int(
                    confusion[
                        class_id, :
                    ].sum()
                ),
        }

    accuracy = (
        np.trace(confusion)
        / max(
            1,
            confusion.sum(),
        )
    )

    return {
        "accuracy":
            float(accuracy),

        "macro_precision":
            float(
                np.mean(
                    precisions
                )
            ),

        "macro_recall":
            float(
                np.mean(
                    recalls
                )
            ),

        "macro_f1":
            float(
                np.mean(
                    f1s
                )
            ),

        "per_class":
            per_class,

        "confusion_matrix":
            confusion.tolist(),
    }


# ---------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------

@torch.no_grad()
def main() -> None:

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("dataset/v3"),
    )

    parser.add_argument(
        "--model",
        type=Path,
        default=Path(
            "outputs/gnn_v5/"
            "best_model_v5.pt"
        ),
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "outputs/gnn_v5/"
            "test_results_v5.json"
        ),
    )

    args = parser.parse_args()

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 70)
    print(
        "TESTING TEMPORAL HIERARCHICAL "
        "GRAPHSAGE + GRU V5"
    )
    print("=" * 70)

    print(
        f"Device: {device}"
    )

    print(
        f"PyTorch: "
        f"{torch.__version__}"
    )

    print()
    print(
        "Loading trained V5 model..."
    )

    checkpoint = torch.load(
    args.model,
    map_location=device,
    weights_only=False,
)

    config = checkpoint[
        "config"
    ]

    hidden_size = config[
        "hidden_size"
    ]

    deadlock_threshold = config[
        "deadlock_threshold"
    ]

    pre_threshold = config[
        "pre_threshold"
    ]

    model = TemporalRiskAwareV5(
        input_size=9,
        hidden_size=hidden_size,
    ).to(device)

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    model.eval()

    print(
        f"Model: {args.model}"
    )

    print(
        f"Hidden size: "
        f"{hidden_size}"
    )

    print(
        "Input size: 9"
    )

    print(
        f"Validation-selected "
        f"DEADLOCK threshold: "
        f"{deadlock_threshold:.2f}"
    )

    print(
        f"Validation-selected "
        f"PRE threshold: "
        f"{pre_threshold:.2f}"
    )

    print(
        "✓ V5 model loaded"
    )

    print()
    print(
        "Loading TEST dataset..."
    )

    dataset = TemporalSequenceDataset(
        args.dataset,
        "test",
    )

    print()
    print(
        f"Test samples: "
        f"{len(dataset):,}"
    )

    targets = np.asarray(
        [
            label
            for _, label, _
            in dataset.sequences
        ],
        dtype=np.int64,
    )

    print()
    print(
        "Test class distribution:"
    )

    for label in LABELS:

        class_id = LABEL_TO_ID[
            label
        ]

        count = np.sum(
            targets == class_id
        )

        print(
            f"  {label:12s}: "
            f"{count:,}"
        )

    all_outputs = {
        key: []
        for key in (
            "deadlock",
            "pre",
            "risk_50",
            "risk_100",
            "risk_300",
        )
    }

    for start in range(
        0,
        len(dataset),
        args.batch_size,
    ):

        end = min(
            start
            + args.batch_size,
            len(dataset),
        )

        batch_graphs = [
            dataset[i][0]
            for i in range(
                start,
                end,
            )
        ]

        outputs = model(
            batch_graphs,
            device,
        )

        for key in all_outputs:

            all_outputs[key].extend(
                torch.sigmoid(
                    outputs[key]
                )
                .cpu()
                .numpy()
                .tolist()
            )

    deadlock_probability = (
        np.asarray(
            all_outputs[
                "deadlock"
            ]
        )
    )

    pre_probability = (
        np.asarray(
            all_outputs["pre"]
        )
    )

    predictions = (
        hierarchical_predictions(
            deadlock_probability,
            pre_probability,
            deadlock_threshold,
            pre_threshold,
        )
    )

    metrics = calculate_metrics(
        predictions,
        targets,
    )

    # -------------------------------------------------------------
    # Risk-head metrics
    # -------------------------------------------------------------

    risk_metrics = {}

    # Reconstruct risk targets from sequence metadata.
    for horizon, key in (
        (
            "50ms",
            "deadlock_within_50ms",
        ),
        (
            "100ms",
            "deadlock_within_100ms",
        ),
        (
            "300ms",
            "deadlock_within_300ms",
        ),
    ):

        risk_targets = []

        for _, label, row in (
            dataset.sequences
        ):

            default_value = (
                label
                != LABEL_TO_ID[
                    "safe"
                ]
            )

            value = row.get(
                "targets",
                {},
            ).get(
                key,
                default_value,
            )

            risk_targets.append(
                int(bool(value))
            )

        risk_targets = np.asarray(
            risk_targets,
            dtype=np.int64,
        )

        risk_probability = (
            np.asarray(
                all_outputs[
                    f"risk_{horizon[:-2]}"
                ]
            )
        )

        risk_predictions = (
            risk_probability >= 0.50
        ).astype(np.int64)

        tp = np.sum(
            (
                risk_predictions == 1
            )
            & (
                risk_targets == 1
            )
        )

        fp = np.sum(
            (
                risk_predictions == 1
            )
            & (
                risk_targets == 0
            )
        )

        fn = np.sum(
            (
                risk_predictions == 0
            )
            & (
                risk_targets == 1
            )
        )

        precision = (
            tp / max(
                1,
                tp + fp,
            )
        )

        recall = (
            tp / max(
                1,
                tp + fn,
            )
        )

        f1 = (
            2
            * precision
            * recall
            / max(
                1e-12,
                precision + recall,
            )
        )

        risk_metrics[horizon] = {
            "precision":
                float(precision),

            "recall":
                float(recall),

            "f1":
                float(f1),
        }

    # -------------------------------------------------------------
    # Print
    # -------------------------------------------------------------

    print()
    print("=" * 70)
    print("FINAL V5 TEST RESULTS")
    print("=" * 70)

    print(
        f"Accuracy:    "
        f"{metrics['accuracy']:.4f}"
    )

    print(
        f"Macro-F1:    "
        f"{metrics['macro_f1']:.4f}"
    )

    print(
        f"Macro-P:     "
        f"{metrics['macro_precision']:.4f}"
    )

    print(
        f"Macro-R:     "
        f"{metrics['macro_recall']:.4f}"
    )

    print()
    print(
        "Per-class results:"
    )

    for label in LABELS:

        item = metrics[
            "per_class"
        ][label]

        print(
            f"  {label:12s} "
            f"P={item['precision']:.4f} "
            f"R={item['recall']:.4f} "
            f"F1={item['f1']:.4f} "
            f"N={item['support']}"
        )

    print()
    print(
        "Confusion Matrix"
    )

    print()
    print(
        "Actual / Predicted       "
        "safe    pre_deadlock    deadlocked"
    )

    confusion = np.asarray(
        metrics[
            "confusion_matrix"
        ]
    )

    for i, label in enumerate(
        LABELS
    ):

        print(
            f"{label:25s}"
            f"{confusion[i, 0]:8d}"
            f"{confusion[i, 1]:16d}"
            f"{confusion[i, 2]:14d}"
        )

    print()
    print(
        "Future-deadlock risk heads:"
    )

    for horizon, item in (
        risk_metrics.items()
    ):

        print(
            f"  {horizon:5s} "
            f"P={item['precision']:.4f} "
            f"R={item['recall']:.4f} "
            f"F1={item['f1']:.4f}"
        )

    # -------------------------------------------------------------
    # Explicit safety checks
    # -------------------------------------------------------------

    safe = metrics[
        "per_class"
    ]["safe"]

    pre = metrics[
        "per_class"
    ]["pre_deadlock"]

    deadlocked = metrics[
        "per_class"
    ]["deadlocked"]

    safe_to_pre = confusion[
        LABEL_TO_ID["safe"],
        LABEL_TO_ID["pre_deadlock"],
    ]

    deadlocked_to_pre = confusion[
        LABEL_TO_ID["deadlocked"],
        LABEL_TO_ID["pre_deadlock"],
    ]

    deadlocked_to_safe = confusion[
        LABEL_TO_ID["deadlocked"],
        LABEL_TO_ID["safe"],
    ]

    print()
    print(
        "=" * 70
    )

    print(
        "V5 SAFETY CHECKS"
    )

    print(
        "=" * 70
    )

    print(
        f"SAFE -> PRE: "
        f"{safe_to_pre:,}"
    )

    print(
        f"DEADLOCKED -> PRE: "
        f"{deadlocked_to_pre:,}"
    )

    print(
        f"DEADLOCKED -> SAFE: "
        f"{deadlocked_to_safe:,}"
    )

    print(
        f"SAFE F1: "
        f"{safe['f1']:.4f}"
    )

    print(
        f"PRE F1: "
        f"{pre['f1']:.4f}"
    )

    print(
        f"DEADLOCKED F1: "
        f"{deadlocked['f1']:.4f}"
    )

    print(
        "=" * 70
    )

    # -------------------------------------------------------------
    # Save
    # -------------------------------------------------------------

    results = {
        "model": str(
            args.model
        ),

        "dataset": str(
            args.dataset
        ),

        "deadlock_threshold":
            float(
                deadlock_threshold
            ),

        "pre_threshold":
            float(
                pre_threshold
            ),

        "metrics":
            metrics,

        "risk_metrics":
            risk_metrics,

        "safety_checks": {
            "safe_to_pre":
                int(safe_to_pre),

            "deadlocked_to_pre":
                int(
                    deadlocked_to_pre
                ),

            "deadlocked_to_safe":
                int(
                    deadlocked_to_safe
                ),
        },
    }

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with args.output.open(
        "w",
        encoding="utf-8",
    ) as handle:

        json.dump(
            results,
            handle,
            indent=2,
        )

    print()
    print(
        f"Results saved to: "
        f"{args.output}"
    )

    print()
    print(
        "V5 TESTING COMPLETE"
    )


if __name__ == "__main__":
    main()
