#!/usr/bin/env python3
"""
Evaluate Temporal Risk-Aware GraphSAGE + GRU V4.

The test set is used ONLY here, after training/model selection.

The checkpoint contains the PRE_DEADLOCK threshold selected using
the validation set.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
from collections import Counter
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

INPUT_SIZE = 9


# ---------------------------------------------------------------------
# File loading
# ---------------------------------------------------------------------

def resolve_jsonl_path(
    dataset_dir: Path,
    filename: str,
) -> Path:

    gz_path = dataset_dir / f"{filename}.gz"
    jsonl_path = dataset_dir / filename

    if gz_path.exists():
        return gz_path

    if jsonl_path.exists():
        return jsonl_path

    raise FileNotFoundError(
        f"Could not find:\n"
        f"{gz_path}\n"
        f"{jsonl_path}"
    )


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

    nodes = snapshot["nodes"]

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

        relation = edge["type"]

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
            (source, target)
        )

    edges = {}

    for relation in RELATIONS:

        pairs = edge_lists[
            relation
        ]

        if pairs:

            edges[relation] = torch.tensor(
                pairs,
                dtype=torch.long,
            ).t().contiguous()

        else:

            edges[relation] = torch.empty(
                (2, 0),
                dtype=torch.long,
            )

    return {
        "x": x,
        "edges": edges,
    }


# ---------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------

class TestDataset:

    def __init__(
        self,
        dataset_dir: Path,
    ):

        snapshot_path = resolve_jsonl_path(
            dataset_dir,
            "test.jsonl",
        )

        sequence_path = resolve_jsonl_path(
            dataset_dir,
            "test_sequences.jsonl",
        )

        print(
            "Loading TEST snapshots..."
        )

        snapshot_rows = load_jsonl(
            snapshot_path
        )

        print(
            "Loading TEST sequences..."
        )

        sequence_rows = load_jsonl(
            sequence_path
        )

        graphs = {
            row["snapshot_id"]:
                tensorize_graph(row)
            for row in snapshot_rows
        }

        self.items = []

        for row in sequence_rows:

            self.items.append(
                {
                    "graphs": [
                        graphs[sid]
                        for sid
                        in row["snapshot_ids"]
                    ],

                    "label":
                        LABEL_TO_ID[
                            row["label"]
                        ],

                    "within_50":
                        float(
                            bool(
                                row.get(
                                    "targets",
                                    {}
                                ).get(
                                    "deadlock_within_50ms",
                                    False,
                                )
                            )
                        ),

                    "within_100":
                        float(
                            bool(
                                row.get(
                                    "targets",
                                    {}
                                ).get(
                                    "deadlock_within_100ms",
                                    False,
                                )
                            )
                        ),

                    "within_300":
                        float(
                            bool(
                                row.get(
                                    "targets",
                                    {}
                                ).get(
                                    "deadlock_within_300ms",
                                    False,
                                )
                            )
                        ),
                }
            )

        print(
            f"Loaded "
            f"{len(snapshot_rows):,} snapshots"
        )

        print(
            f"Loaded "
            f"{len(self.items):,} sequences"
        )

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]


# ---------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------

class RelationalGraphSAGE(nn.Module):

    def __init__(
        self,
        input_size=INPUT_SIZE,
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

        self.relation_layers = nn.ModuleDict(
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

            out = out + self.aggregate(
                h,
                edges[relation],
                self.relation_layers[
                    relation
                ],
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


class TemporalRiskAwareGraphClassifier(
    nn.Module
):

    def __init__(
        self,
        input_size=INPUT_SIZE,
        hidden_size=32,
    ):

        super().__init__()

        self.gnn = RelationalGraphSAGE(
            input_size=input_size,
            hidden_size=hidden_size,
        )

        self.gru = nn.GRU(
            input_size=hidden_size * 2,
            hidden_size=hidden_size,
            batch_first=True,
        )

        self.shared_norm = nn.LayerNorm(
            hidden_size
        )

        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(
                hidden_size,
                3,
            ),
        )

        self.risk_50 = nn.Linear(
            hidden_size,
            1,
        )

        self.risk_100 = nn.Linear(
            hidden_size,
            1,
        )

        self.risk_300 = nn.Linear(
            hidden_size,
            1,
        )

        self.risk_score = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(
                hidden_size,
                1,
            ),
        )

    def forward(
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
                            edge.to(device)
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

        temporal = self.shared_norm(
            output[:, -1, :]
        )

        return {
            "logits":
                self.classifier(
                    temporal
                ),

            "risk_50":
                self.risk_50(
                    temporal
                ).squeeze(1),

            "risk_100":
                self.risk_100(
                    temporal
                ).squeeze(1),

            "risk_300":
                self.risk_300(
                    temporal
                ).squeeze(1),

            "risk_score":
                self.risk_score(
                    temporal
                ).squeeze(1),
        }


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
            prediction
        ] += 1

    per_class = {}

    precision_values = []
    recall_values = []
    f1_values = []

    for class_id, label in enumerate(
        LABELS
    ):

        tp = confusion[
            class_id,
            class_id
        ]

        fp = (
            confusion[
                :,
                class_id
            ].sum()
            - tp
        )

        fn = (
            confusion[
                class_id,
                :
            ].sum()
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

        precision_values.append(
            precision
        )

        recall_values.append(
            recall
        )

        f1_values.append(
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
                        class_id,
                        :
                    ].sum()
                ),
        }

    return {
        "accuracy":
            float(
                np.trace(confusion)
                / max(
                    1,
                    confusion.sum(),
                )
            ),

        "macro_precision":
            float(
                np.mean(
                    precision_values
                )
            ),

        "macro_recall":
            float(
                np.mean(
                    recall_values
                )
            ),

        "macro_f1":
            float(
                np.mean(
                    f1_values
                )
            ),

        "per_class":
            per_class,

        "confusion_matrix":
            confusion.tolist(),
    }


# ---------------------------------------------------------------------
# Prediction rule
# ---------------------------------------------------------------------

def apply_pre_threshold(
    probabilities,
    threshold,
):

    predictions = probabilities.argmax(
        axis=1
    )

    safe_id = LABEL_TO_ID[
        "safe"
    ]

    pre_id = LABEL_TO_ID[
        "pre_deadlock"
    ]

    dead_id = LABEL_TO_ID[
        "deadlocked"
    ]

    for i in range(
        len(predictions)
    ):

        p_safe = probabilities[
            i,
            safe_id,
        ]

        p_pre = probabilities[
            i,
            pre_id,
        ]

        p_dead = probabilities[
            i,
            dead_id,
        ]

        # Preserve a deadlock prediction
        # if DEADLOCK is the strongest class.
        if (
            p_dead >= p_safe
            and p_dead >= p_pre
        ):

            predictions[i] = (
                dead_id
            )

        elif (
            p_pre >= threshold
            and p_pre > p_safe
        ):

            predictions[i] = (
                pre_id
            )

        else:

            predictions[i] = (
                safe_id
            )

    return predictions


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path(
            "dataset/v3"
        ),
    )

    parser.add_argument(
        "--model",
        type=Path,
        default=Path(
            "outputs/gnn_v4/"
            "best_model_v4.pt"
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
            "outputs/gnn_v4/"
            "test_results_v4.json"
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
        "TESTING TEMPORAL RISK-AWARE "
        "GRAPHSAGE + GRU V4"
    )
    print("=" * 70)

    print(
        f"Device: {device}"
    )

    print(
        f"PyTorch: "
        f"{torch.__version__}"
    )

    # ---------------------------------------------------------------
    # Load checkpoint
    # ---------------------------------------------------------------

    print()
    print(
        "Loading trained V4 model..."
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

    input_size = config.get(
        "input_size",
        INPUT_SIZE,
    )

    pre_threshold = config.get(
        "pre_threshold",
        0.50,
    )

    print(
        f"Model: {args.model}"
    )

    print(
        f"Hidden size: "
        f"{hidden_size}"
    )

    print(
        f"Input size: "
        f"{input_size}"
    )

    print(
        f"Validation-selected "
        f"PRE threshold: "
        f"{pre_threshold:.2f}"
    )

    model = (
        TemporalRiskAwareGraphClassifier(
            input_size=input_size,
            hidden_size=hidden_size,
        ).to(device)
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    model.eval()

    print(
        "✓ V4 model loaded"
    )

    # ---------------------------------------------------------------
    # Test dataset
    # ---------------------------------------------------------------

    print()
    print(
        "Loading TEST dataset..."
    )

    dataset = TestDataset(
        args.dataset
    )

    print()
    print(
        f"Test samples: "
        f"{len(dataset):,}"
    )

    counts = Counter(
        item["label"]
        for item in dataset.items
    )

    print()
    print(
        "Test class distribution:"
    )

    for label in LABELS:

        print(
            f"  {label:14s}: "
            f"{counts.get(LABEL_TO_ID[label], 0):,}"
        )

    # ---------------------------------------------------------------
    # Inference
    # ---------------------------------------------------------------

    probabilities_list = []

    targets = []

    risk50_targets = []
    risk100_targets = []
    risk300_targets = []

    risk50_predictions = []
    risk100_predictions = []
    risk300_predictions = []

    with torch.no_grad():

        for start in range(
            0,
            len(dataset),
            args.batch_size,
        ):

            batch_items = [
                dataset[i]
                for i
                in range(
                    start,
                    min(
                        start
                        + args.batch_size,
                        len(dataset),
                    ),
                )
            ]

            batch_graphs = [
                item["graphs"]
                for item
                in batch_items
            ]

            batch_targets = [
                item["label"]
                for item in batch_items
            ]

            outputs = model(
                batch_graphs,
                device,
            )

            probabilities = (
                torch.softmax(
                    outputs["logits"],
                    dim=1,
                )
                .cpu()
                .numpy()
            )

            probabilities_list.append(
                probabilities
            )

            targets.extend(
                batch_targets
            )

            risk50_targets.extend(
                item["within_50"]
                for item in batch_items
            )

            risk100_targets.extend(
                item["within_100"]
                for item in batch_items
            )

            risk300_targets.extend(
                item["within_300"]
                for item in batch_items
            )

            risk50_predictions.extend(
                (
                    torch.sigmoid(
                        outputs["risk_50"]
                    )
                    >= 0.5
                )
                .long()
                .cpu()
                .tolist()
            )

            risk100_predictions.extend(
                (
                    torch.sigmoid(
                        outputs["risk_100"]
                    )
                    >= 0.5
                )
                .long()
                .cpu()
                .tolist()
            )

            risk300_predictions.extend(
                (
                    torch.sigmoid(
                        outputs["risk_300"]
                    )
                    >= 0.5
                )
                .long()
                .cpu()
                .tolist()
            )

    probabilities = np.concatenate(
        probabilities_list,
        axis=0,
    )

    # ---------------------------------------------------------------
    # Standard argmax
    # ---------------------------------------------------------------

    argmax_predictions = (
        probabilities.argmax(
            axis=1
        )
    )

    argmax_metrics = calculate_metrics(
        argmax_predictions.tolist(),
        targets,
    )

    # ---------------------------------------------------------------
    # Validation-selected threshold
    # ---------------------------------------------------------------

    tuned_predictions = (
        apply_pre_threshold(
            probabilities,
            pre_threshold,
        )
    )

    tuned_metrics = calculate_metrics(
        tuned_predictions.tolist(),
        targets,
    )

    # ---------------------------------------------------------------
    # Risk-head metrics
    # ---------------------------------------------------------------

    def binary_metrics(
        predictions,
        targets,
    ):

        predictions = np.asarray(
            predictions
        )

        targets = np.asarray(
            targets
        )

        tp = int(
            np.sum(
                (predictions == 1)
                & (targets == 1)
            )
        )

        fp = int(
            np.sum(
                (predictions == 1)
                & (targets == 0)
            )
        )

        fn = int(
            np.sum(
                (predictions == 0)
                & (targets == 1)
            )
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

        return {
            "precision":
                float(precision),

            "recall":
                float(recall),

            "f1":
                float(f1),

            "tp": tp,
            "fp": fp,
            "fn": fn,
        }

    risk_metrics = {
        "50ms": binary_metrics(
            risk50_predictions,
            risk50_targets,
        ),

        "100ms": binary_metrics(
            risk100_predictions,
            risk100_targets,
        ),

        "300ms": binary_metrics(
            risk300_predictions,
            risk300_targets,
        ),
    }

    # ---------------------------------------------------------------
    # Print
    # ---------------------------------------------------------------

    print()
    print("=" * 70)
    print(
        "FINAL V4 TEST RESULTS"
    )
    print("=" * 70)

    print()
    print(
        "STANDARD ARGMAX"
    )

    print(
        f"Accuracy:    "
        f"{argmax_metrics['accuracy']:.4f}"
    )

    print(
        f"Macro-F1:    "
        f"{argmax_metrics['macro_f1']:.4f}"
    )

    print()
    print(
        "VALIDATION-TUNED PRE THRESHOLD"
    )

    print(
        f"PRE threshold: "
        f"{pre_threshold:.2f}"
    )

    print(
        f"Accuracy:    "
        f"{tuned_metrics['accuracy']:.4f}"
    )

    print(
        f"Macro-F1:    "
        f"{tuned_metrics['macro_f1']:.4f}"
    )

    print(
        f"Macro-P:     "
        f"{tuned_metrics['macro_precision']:.4f}"
    )

    print(
        f"Macro-R:     "
        f"{tuned_metrics['macro_recall']:.4f}"
    )

    print()
    print(
        "Per-class results:"
    )

    for label in LABELS:

        item = tuned_metrics[
            "per_class"
        ][label]

        print(
            f"  {label:14s} "
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

    matrix = tuned_metrics[
        "confusion_matrix"
    ]

    for label, row in zip(
        LABELS,
        matrix,
    ):

        print(
            f"{label:23s}"
            f"{row[0]:8d}"
            f"{row[1]:16d}"
            f"{row[2]:14d}"
        )

    print()
    print(
        "Future-deadlock risk heads:"
    )

    for horizon in (
        "50ms",
        "100ms",
        "300ms",
    ):

        item = risk_metrics[
            horizon
        ]

        print(
            f"  {horizon:6s} "
            f"P={item['precision']:.4f} "
            f"R={item['recall']:.4f} "
            f"F1={item['f1']:.4f}"
        )

    # ---------------------------------------------------------------
    # Save
    # ---------------------------------------------------------------

    results = {
        "model": str(
            args.model
        ),

        "dataset": str(
            args.dataset
        ),

        "model_version": "V4",

        "pre_threshold":
            pre_threshold,

        "standard_argmax":
            argmax_metrics,

        "tuned":
            tuned_metrics,

        "risk_heads":
            risk_metrics,

        "checkpoint_epoch":
            checkpoint.get(
                "epoch"
            ),

        "validation_selection_score":
            checkpoint.get(
                "best_selection_score"
            ),
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
    print("=" * 70)
    print(
        "V4 TESTING COMPLETE"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()
