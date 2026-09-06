#!/usr/bin/env python3
"""
V5 Deadlock Prediction Model

Temporal Risk-Aware Relational GraphSAGE + GRU

V5 changes:
1. Hierarchical prediction instead of a single 3-class softmax.
2. Dedicated CURRENT-DEADLOCK binary head.
3. Dedicated PRE-DEADLOCK binary head.
4. 300 ms future-risk auxiliary head.
5. PRE is evaluated only when the current-deadlock head says
   the system is NOT currently deadlocked.
6. Validation threshold selection explicitly protects SAFE precision
   and DEADLOCK precision.
7. Test set is never used during training or threshold selection.
8. Supports .jsonl.gz and .jsonl files.

Decision hierarchy:

    current_deadlock probability >= deadlock_threshold
        -> DEADLOCKED

    otherwise:
        pre_deadlock probability >= pre_threshold
        -> PRE_DEADLOCK

    otherwise:
        -> SAFE
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import random
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


LABELS = ("safe", "pre_deadlock", "deadlocked")
LABEL_TO_ID = {label: i for i, label in enumerate(LABELS)}

RELATIONS = ("owned_by", "waits_for")


# ---------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------
# JSONL loading
# ---------------------------------------------------------------------

def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    if path.suffix == ".gz":
        handle = gzip.open(path, "rt", encoding="utf-8")
    else:
        handle = path.open("r", encoding="utf-8")

    with handle:
        for line in handle:
            line = line.strip()

            if line:
                rows.append(json.loads(line))

    return rows


def resolve_jsonl_path(dataset_dir: Path, filename: str) -> Path:
    compressed = dataset_dir / f"{filename}.gz"
    plain = dataset_dir / filename

    if compressed.exists():
        return compressed

    if plain.exists():
        return plain

    raise FileNotFoundError(
        f"Could not find either {compressed} or {plain}"
    )


# ---------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------

def node_features(node: dict[str, Any]) -> list[float]:
    """
    Keep the same 9-D feature contract used by V3/V4.

    0: is_thread
    1: is_lock
    2: is_waiting
    3: log1p(wait_ns) / 22
    4: has_owner
    5: log1p(scheduler_switches) / 8
    6: log1p(wakeups) / 8
    7: log1p(cpu_migrations) / 6
    8: normalized last_cpu
    """

    node_type = node.get("type", "")
    features = node.get("features", {})

    return [
        float(node_type == "thread"),
        float(node_type == "lock"),
        float(features.get("is_waiting", 0)),
        math.log1p(float(features.get("wait_ns", 0))) / 22.0,
        float(features.get("has_owner", 0)),
        math.log1p(float(features.get("scheduler_switches", 0))) / 8.0,
        math.log1p(float(features.get("wakeups", 0))) / 8.0,
        math.log1p(float(features.get("cpu_migrations", 0))) / 6.0,
        float(features.get("last_cpu", 0)) / 7.0,
    ]


def tensorize_graph(snapshot: dict[str, Any]) -> dict[str, Any]:
    nodes = snapshot["nodes"]

    node_index = {
        node["id"]: i
        for i, node in enumerate(nodes)
    }

    x = torch.tensor(
        [node_features(node) for node in nodes],
        dtype=torch.float32,
    )

    edge_lists: dict[str, list[tuple[int, int]]] = {
        relation: []
        for relation in RELATIONS
    }

    for edge in snapshot.get("edges", []):
        relation = edge["type"]

        if relation not in edge_lists:
            continue

        source = node_index[edge["source"]]
        target = node_index[edge["target"]]

        edge_lists[relation].append(
            (source, target)
        )

    edges: dict[str, torch.Tensor] = {}

    for relation in RELATIONS:
        pairs = edge_lists[relation]

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

class TemporalSequenceDataset:
    def __init__(
        self,
        dataset_dir: Path,
        split: str,
    ):
        self.dataset_dir = dataset_dir
        self.split = split

        print(f"Loading {split.upper()} snapshots...")
        snapshot_rows = load_jsonl(
            resolve_jsonl_path(
                dataset_dir,
                f"{split}.jsonl",
            )
        )

        print(f"Loading {split.upper()} sequences...")
        sequence_rows = load_jsonl(
            resolve_jsonl_path(
                dataset_dir,
                f"{split}_sequences.jsonl",
            )
        )

        self.graphs = {
            row["snapshot_id"]: tensorize_graph(row)
            for row in snapshot_rows
        }

        self.sequences: list[
            tuple[list[dict[str, Any]], int, dict[str, Any]]
        ] = []

        for row in sequence_rows:
            graphs = [
                self.graphs[sid]
                for sid in row["snapshot_ids"]
            ]

            label = LABEL_TO_ID[row["label"]]

            self.sequences.append(
                (
                    graphs,
                    label,
                    row,
                )
            )

        self.labels = [
            label
            for _, label, _ in self.sequences
        ]

        print(
            f"Loaded {len(snapshot_rows):,} snapshots"
        )

        print(
            f"Loaded {len(self.sequences):,} sequences"
        )

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(
        self,
        index: int,
    ):
        return self.sequences[index]

    def label_counts(self) -> Counter:
        return Counter(self.labels)


# ---------------------------------------------------------------------
# Relational GraphSAGE
# ---------------------------------------------------------------------

class RelationalGraphSAGE(nn.Module):

    def __init__(
        self,
        input_size: int = 9,
        hidden_size: int = 32,
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
                relation: nn.Linear(
                    hidden_size,
                    hidden_size,
                    bias=False,
                )
                for relation in RELATIONS
            }
        )

        self.norm = nn.LayerNorm(
            hidden_size
        )

    def aggregate(
        self,
        h: torch.Tensor,
        edge_index: torch.Tensor,
        transform: nn.Module,
    ) -> torch.Tensor:

        if edge_index.numel() == 0:
            return torch.zeros_like(h)

        source = edge_index[0]
        target = edge_index[1]

        messages = transform(
            h[source]
        )

        aggregated = torch.zeros_like(h)

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

        return aggregated / counts

    def forward(
        self,
        graph: dict[str, Any],
    ) -> torch.Tensor:

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
                self.relation_layers[relation],
            )

        h = F.relu(
            self.norm(out)
        )

        thread_mask = x[:, 0] > 0.5
        lock_mask = x[:, 1] > 0.5

        if thread_mask.any():
            thread_pool = h[
                thread_mask
            ].mean(dim=0)
        else:
            thread_pool = h.mean(dim=0)

        if lock_mask.any():
            lock_pool = h[
                lock_mask
            ].mean(dim=0)
        else:
            lock_pool = h.mean(dim=0)

        return torch.cat(
            [
                thread_pool,
                lock_pool,
            ],
            dim=0,
        )


# ---------------------------------------------------------------------
# V5 hierarchical temporal model
# ---------------------------------------------------------------------

class TemporalRiskAwareV5(nn.Module):

    def __init__(
        self,
        input_size: int = 9,
        hidden_size: int = 32,
    ):
        super().__init__()

        self.hidden_size = hidden_size

        self.gnn = RelationalGraphSAGE(
            input_size=input_size,
            hidden_size=hidden_size,
        )

        self.gru = nn.GRU(
            input_size=hidden_size * 2,
            hidden_size=hidden_size,
            batch_first=True,
        )

        self.norm = nn.LayerNorm(
            hidden_size
        )

        # Current deadlock probability.
        self.deadlock_head = nn.Linear(
            hidden_size,
            1,
        )

        # PRE-DEADLOCK probability.
        self.pre_head = nn.Linear(
            hidden_size,
            1,
        )

        # Future risk auxiliary heads.
        self.risk_50_head = nn.Linear(
            hidden_size,
            1,
        )

        self.risk_100_head = nn.Linear(
            hidden_size,
            1,
        )

        self.risk_300_head = nn.Linear(
            hidden_size,
            1,
        )

    def encode(
        self,
        batch_graph_sequences: list[
            list[dict[str, Any]]
        ],
        device: torch.device,
    ) -> torch.Tensor:

        sequence_embeddings = []

        for graphs in batch_graph_sequences:

            embeddings = []

            for graph in graphs:

                graph = {
                    "x": graph["x"].to(device),
                    "edges": {
                        relation: edge.to(device)
                        for relation, edge
                        in graph["edges"].items()
                    },
                }

                embeddings.append(
                    self.gnn(graph)
                )

            sequence_embeddings.append(
                torch.stack(embeddings)
            )

        x = torch.stack(
            sequence_embeddings,
            dim=0,
        )

        output, _ = self.gru(x)

        return self.norm(
            output[:, -1, :]
        )

    def forward(
        self,
        batch_graph_sequences: list[
            list[dict[str, Any]]
        ],
        device: torch.device,
    ) -> dict[str, torch.Tensor]:

        h = self.encode(
            batch_graph_sequences,
            device,
        )

        return {
            "deadlock": self.deadlock_head(h).squeeze(1),
            "pre": self.pre_head(h).squeeze(1),
            "risk_50": self.risk_50_head(h).squeeze(1),
            "risk_100": self.risk_100_head(h).squeeze(1),
            "risk_300": self.risk_300_head(h).squeeze(1),
        }


# ---------------------------------------------------------------------
# Binary focal loss
# ---------------------------------------------------------------------

class BinaryFocalLoss(nn.Module):

    def __init__(
        self,
        pos_weight: float = 1.0,
        gamma: float = 1.0,
    ):
        super().__init__()

        self.pos_weight = pos_weight
        self.gamma = gamma

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:

        targets = targets.float()

        probabilities = torch.sigmoid(
            logits
        )

        bce = F.binary_cross_entropy_with_logits(
            logits,
            targets,
            pos_weight=torch.tensor(
                self.pos_weight,
                device=logits.device,
            ),
            reduction="none",
        )

        p_t = (
            probabilities * targets
            + (1.0 - probabilities)
            * (1.0 - targets)
        )

        focal_factor = (
            1.0 - p_t
        ).pow(self.gamma)

        return (
            focal_factor * bce
        ).mean()


# ---------------------------------------------------------------------
# Target extraction
# ---------------------------------------------------------------------

def get_targets(
    rows: list[dict[str, Any]],
) -> dict[str, torch.Tensor]:

    labels = [
        LABEL_TO_ID[row["label"]]
        for row in rows
    ]

    deadlock = [
        float(label == LABEL_TO_ID["deadlocked"])
        for label in labels
    ]

    pre = [
        float(label == LABEL_TO_ID["pre_deadlock"])
        for label in labels
    ]

    risk_50 = [
        float(
            row.get(
                "targets",
                {},
            ).get(
                "deadlock_within_50ms",
                label == LABEL_TO_ID["deadlocked"],
            )
        )
        for row, label in zip(rows, labels)
    ]

    risk_100 = [
        float(
            row.get(
                "targets",
                {},
            ).get(
                "deadlock_within_100ms",
                label != LABEL_TO_ID["safe"],
            )
        )
        for row, label in zip(rows, labels)
    ]

    risk_300 = [
        float(
            row.get(
                "targets",
                {},
            ).get(
                "deadlock_within_300ms",
                label != LABEL_TO_ID["safe"],
            )
        )
        for row, label in zip(rows, labels)
    ]

    return {
        "deadlock": torch.tensor(
            deadlock,
            dtype=torch.float32,
        ),
        "pre": torch.tensor(
            pre,
            dtype=torch.float32,
        ),
        "risk_50": torch.tensor(
            risk_50,
            dtype=torch.float32,
        ),
        "risk_100": torch.tensor(
            risk_100,
            dtype=torch.float32,
        ),
        "risk_300": torch.tensor(
            risk_300,
            dtype=torch.float32,
        ),
    }


# ---------------------------------------------------------------------
# Hierarchical predictions
# ---------------------------------------------------------------------

def hierarchical_predictions(
    deadlock_probability: np.ndarray,
    pre_probability: np.ndarray,
    deadlock_threshold: float,
    pre_threshold: float,
) -> np.ndarray:

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
    ] = LABEL_TO_ID["deadlocked"]

    non_deadlocked = ~deadlocked_mask

    pre_mask = (
        pre_probability
        >= pre_threshold
    )

    predictions[
        non_deadlocked & pre_mask
    ] = LABEL_TO_ID["pre_deadlock"]

    predictions[
        non_deadlocked & ~pre_mask
    ] = LABEL_TO_ID["safe"]

    return predictions


# ---------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------

def calculate_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
) -> dict[str, Any]:

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
            confusion[:, class_id].sum()
            - tp
        )

        fn = (
            confusion[class_id, :].sum()
            - tp
        )

        precision = (
            tp / max(1, tp + fp)
        )

        recall = (
            tp / max(1, tp + fn)
        )

        f1 = (
            2.0
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
            "precision": float(
                precision
            ),
            "recall": float(
                recall
            ),
            "f1": float(f1),
            "support": int(
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
        "accuracy": float(
            accuracy
        ),
        "macro_precision": float(
            np.mean(precisions)
        ),
        "macro_recall": float(
            np.mean(recalls)
        ),
        "macro_f1": float(
            np.mean(f1s)
        ),
        "per_class": per_class,
        "confusion_matrix":
            confusion.tolist(),
    }


# ---------------------------------------------------------------------
# Validation threshold search
# ---------------------------------------------------------------------

def tune_thresholds(
    deadlock_prob: np.ndarray,
    pre_prob: np.ndarray,
    targets: np.ndarray,
) -> dict[str, Any]:

    best = None

    # We deliberately search conservative deadlock thresholds.
    deadlock_thresholds = np.arange(
        0.50,
        0.91,
        0.02,
    )

    pre_thresholds = np.arange(
        0.40,
        0.91,
        0.02,
    )

    for deadlock_threshold in deadlock_thresholds:

        for pre_threshold in pre_thresholds:

            predictions = hierarchical_predictions(
                deadlock_prob,
                pre_prob,
                float(deadlock_threshold),
                float(pre_threshold),
            )

            metrics = calculate_metrics(
                predictions,
                targets,
            )

            safe = metrics[
                "per_class"
            ]["safe"]

            pre = metrics[
                "per_class"
            ]["pre_deadlock"]

            deadlocked = metrics[
                "per_class"
            ]["deadlocked"]

            # ---------------------------------------------------------
            # Safety constraints
            # ---------------------------------------------------------
            #
            # SAFE precision >= 0.80
            # DEADLOCK precision >= 0.98
            #
            # These prevent V5 from "solving" PRE by simply calling
            # everything PRE or DEADLOCK.
            #
            if (
                safe["precision"] < 0.80
                or deadlocked["precision"] < 0.98
            ):
                continue

            # Main objective:
            # PRE F1 is important, but safe/deadlocked performance
            # remains part of the selection score.
            score = (
                0.50 * pre["f1"]
                + 0.25 * safe["f1"]
                + 0.25 * deadlocked["f1"]
            )

            candidate = {
                "score": float(score),
                "deadlock_threshold":
                    float(deadlock_threshold),
                "pre_threshold":
                    float(pre_threshold),
                "metrics": metrics,
            }

            if (
                best is None
                or candidate["score"]
                > best["score"]
            ):
                best = candidate

    # If the strict constraints cannot be satisfied,
    # choose the best unconstrained configuration.
    if best is None:

        for deadlock_threshold in deadlock_thresholds:

            for pre_threshold in pre_thresholds:

                predictions = hierarchical_predictions(
                    deadlock_prob,
                    pre_prob,
                    float(deadlock_threshold),
                    float(pre_threshold),
                )

                metrics = calculate_metrics(
                    predictions,
                    targets,
                )

                safe = metrics[
                    "per_class"
                ]["safe"]

                pre = metrics[
                    "per_class"
                ]["pre_deadlock"]

                deadlocked = metrics[
                    "per_class"
                ]["deadlocked"]

                score = (
                    0.50 * pre["f1"]
                    + 0.25 * safe["f1"]
                    + 0.25 * deadlocked["f1"]
                )

                candidate = {
                    "score": float(score),
                    "deadlock_threshold":
                        float(deadlock_threshold),
                    "pre_threshold":
                        float(pre_threshold),
                    "metrics": metrics,
                }

                if (
                    best is None
                    or candidate["score"]
                    > best["score"]
                ):
                    best = candidate

    return best


# ---------------------------------------------------------------------
# One epoch
# ---------------------------------------------------------------------

def run_epoch(
    model: TemporalRiskAwareV5,
    dataset: TemporalSequenceDataset,
    optimizer: torch.optim.Optimizer | None,
    losses: dict[str, nn.Module],
    device: torch.device,
    batch_size: int,
    train: bool,
    seed: int,
) -> dict[str, Any]:

    model.train(train)

    indices = list(
        range(len(dataset))
    )

    if train:
        random.Random(seed).shuffle(
            indices
        )

    total_loss = 0.0
    total_samples = 0

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

    all_targets = {
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
        len(indices),
        batch_size,
    ):

        batch_indices = indices[
            start:start + batch_size
        ]

        batch_graphs = [
            dataset[i][0]
            for i in batch_indices
        ]

        batch_rows = [
            dataset[i][2]
            for i in batch_indices
        ]

        targets = get_targets(
            batch_rows
        )

        targets = {
            key: value.to(device)
            for key, value in targets.items()
        }

        if train:
            optimizer.zero_grad(
                set_to_none=True
            )

        with torch.set_grad_enabled(
            train
        ):

            outputs = model(
                batch_graphs,
                device,
            )

            # Main losses.
            deadlock_loss = losses[
                "deadlock"
            ](
                outputs["deadlock"],
                targets["deadlock"],
            )

            pre_loss = losses[
                "pre"
            ](
                outputs["pre"],
                targets["pre"],
            )

            # Auxiliary temporal-risk losses.
            risk_50_loss = losses[
                "risk_50"
            ](
                outputs["risk_50"],
                targets["risk_50"],
            )

            risk_100_loss = losses[
                "risk_100"
            ](
                outputs["risk_100"],
                targets["risk_100"],
            )

            risk_300_loss = losses[
                "risk_300"
            ](
                outputs["risk_300"],
                targets["risk_300"],
            )

            # Main objective:
            #
            # Deadlock and PRE classification dominate.
            # Risk heads provide temporal supervision.
            loss = (
                1.00 * deadlock_loss
                + 1.50 * pre_loss
                + 0.20 * risk_50_loss
                + 0.30 * risk_100_loss
                + 0.70 * risk_300_loss
            )

            if train:

                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=1.0,
                )

                optimizer.step()

        batch_count = len(
            batch_indices
        )

        total_loss += (
            float(loss.item())
            * batch_count
        )

        total_samples += batch_count

        for key in all_outputs:

            all_outputs[key].extend(
                torch.sigmoid(
                    outputs[key]
                )
                .detach()
                .cpu()
                .numpy()
                .tolist()
            )

            all_targets[key].extend(
                targets[key]
                .detach()
                .cpu()
                .numpy()
                .tolist()
            )

    average_loss = (
        total_loss
        / max(
            1,
            total_samples,
        )
    )

    deadlock_prob = np.asarray(
        all_outputs["deadlock"]
    )

    pre_prob = np.asarray(
        all_outputs["pre"]
    )

    targets_class = np.asarray(
    [
        row[1]
        for row in dataset.sequences
    ],
    dtype=np.int64,
)

    # Threshold tuning is performed only on validation.
    if train:

        deadlock_threshold = 0.50
        pre_threshold = 0.50

    else:

        tuned = tune_thresholds(
            deadlock_prob,
            pre_prob,
            targets_class,
        )

        deadlock_threshold = tuned[
            "deadlock_threshold"
        ]

        pre_threshold = tuned[
            "pre_threshold"
        ]

    predictions = hierarchical_predictions(
        deadlock_prob,
        pre_prob,
        deadlock_threshold,
        pre_threshold,
    )

    metrics = calculate_metrics(
        predictions,
        targets_class,
    )

    # Auxiliary risk metrics.
    risk_metrics = {}

    for horizon in (
        "risk_50",
        "risk_100",
        "risk_300",
    ):

        risk_predictions = (
            np.asarray(
                all_outputs[horizon]
            )
            >= 0.50
        ).astype(np.int64)

        risk_targets = np.asarray(
            all_targets[horizon]
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
            tp / max(1, tp + fp)
        )

        recall = (
            tp / max(1, tp + fn)
        )

        f1 = (
            2 * precision * recall
            / max(
                1e-12,
                precision + recall,
            )
        )

        risk_metrics[horizon] = {
            "precision": float(
                precision
            ),
            "recall": float(
                recall
            ),
            "f1": float(f1),
        }

    return {
        "loss": float(
            average_loss
        ),
        "metrics": metrics,
        "risk_metrics": risk_metrics,
        "deadlock_threshold":
            float(deadlock_threshold),
        "pre_threshold":
            float(pre_threshold),
    }


# ---------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------

def print_results(
    prefix: str,
    result: dict[str, Any],
) -> None:

    metrics = result["metrics"]

    print(
        f"{prefix} loss: "
        f"{result['loss']:.4f} | "
        f"accuracy: "
        f"{metrics['accuracy']:.4f} | "
        f"macro-F1: "
        f"{metrics['macro_f1']:.4f}"
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


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("dataset/v3"),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/gnn_v5"),
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--hidden-size",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    args = parser.parse_args()

    set_seed(args.seed)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 70)
    print(
        "TRAINING TEMPORAL HIERARCHICAL "
        "GRAPHSAGE + GRU V5"
    )
    print("=" * 70)

    print(f"Device: {device}")
    print(f"PyTorch: {torch.__version__}")
    print(f"Dataset: {args.dataset}")
    print(f"Output: {args.output}")

    if torch.cuda.is_available():

        print(
            f"GPU: "
            f"{torch.cuda.get_device_name(0)}"
        )

    else:

        print("GPU: CPU only")

    train_dataset = TemporalSequenceDataset(
        args.dataset,
        "train",
    )

    validation_dataset = TemporalSequenceDataset(
        args.dataset,
        "validation",
    )

    print()
    print(
        f"Training samples: "
        f"{len(train_dataset):,}"
    )

    print(
        f"Validation samples: "
        f"{len(validation_dataset):,}"
    )

    print(
        "Training class distribution:",
        dict(
            train_dataset.label_counts()
        ),
    )

    print(
        "Validation class distribution:",
        dict(
            validation_dataset.label_counts()
        ),
    )

    model = TemporalRiskAwareV5(
        input_size=9,
        hidden_size=args.hidden_size,
    ).to(device)

    print(
        f"Model parameters: "
        f"{sum(p.numel() for p in model.parameters()):,}"
    )

    # -------------------------------------------------------------
    # Positive weights
    # -------------------------------------------------------------

    train_labels = np.asarray(
        train_dataset.labels
    )

    deadlock_positive = np.sum(
        train_labels
        == LABEL_TO_ID["deadlocked"]
    )

    pre_positive = np.sum(
        train_labels
        == LABEL_TO_ID["pre_deadlock"]
    )

    total = len(train_labels)

    deadlock_pos_weight = (
        (total - deadlock_positive)
        / max(1, deadlock_positive)
    )

    pre_pos_weight = (
        (total - pre_positive)
        / max(1, pre_positive)
    )

    print()
    print(
        f"Deadlock positive weight: "
        f"{deadlock_pos_weight:.4f}"
    )

    print(
        f"PRE positive weight: "
        f"{pre_pos_weight:.4f}"
    )

    losses = {
        "deadlock": BinaryFocalLoss(
            pos_weight=deadlock_pos_weight,
            gamma=1.0,
        ).to(device),

        "pre": BinaryFocalLoss(
            pos_weight=pre_pos_weight,
            gamma=1.0,
        ).to(device),

        "risk_50": BinaryFocalLoss(
            pos_weight=1.0,
            gamma=1.0,
        ).to(device),

        "risk_100": BinaryFocalLoss(
            pos_weight=1.0,
            gamma=1.0,
        ).to(device),

        "risk_300": BinaryFocalLoss(
            pos_weight=1.0,
            gamma=1.0,
        ).to(device),
    }

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    args.output.mkdir(
        parents=True,
        exist_ok=True,
    )

    best_score = -1.0
    best_epoch = -1
    epochs_without_improvement = 0

    history = []

    for epoch in range(
        1,
        args.epochs + 1,
    ):

        print()
        print("-" * 70)
        print(
            f"Epoch {epoch}/{args.epochs}"
        )

        start_time = time.time()

        train_result = run_epoch(
            model=model,
            dataset=train_dataset,
            optimizer=optimizer,
            losses=losses,
            device=device,
            batch_size=args.batch_size,
            train=True,
            seed=args.seed + epoch,
        )

        validation_result = run_epoch(
            model=model,
            dataset=validation_dataset,
            optimizer=None,
            losses=losses,
            device=device,
            batch_size=args.batch_size,
            train=False,
            seed=args.seed,
        )

        elapsed = (
            time.time()
            - start_time
        )

        print_results(
            "Train",
            train_result,
        )

        print_results(
            "Validation",
            validation_result,
        )

        print()
        print(
            "Validation thresholds:"
        )

        print(
            f"  DEADLOCK threshold: "
            f"{validation_result['deadlock_threshold']:.2f}"
        )

        print(
            f"  PRE threshold: "
            f"{validation_result['pre_threshold']:.2f}"
        )

        print()
        print(
            "Future-risk heads:"
        )

        for horizon, item in (
            validation_result[
                "risk_metrics"
            ].items()
        ):

            print(
                f"  {horizon:7s} "
                f"P={item['precision']:.4f} "
                f"R={item['recall']:.4f} "
                f"F1={item['f1']:.4f}"
            )

        metrics = (
            validation_result["metrics"]
        )

        safe = metrics[
            "per_class"
        ]["safe"]

        pre = metrics[
            "per_class"
        ]["pre_deadlock"]

        deadlocked = metrics[
            "per_class"
        ]["deadlocked"]

        selection_score = (
            0.50 * pre["f1"]
            + 0.25 * safe["f1"]
            + 0.25 * deadlocked["f1"]
        )

        print()
        print(
            f"Selection score: "
            f"{selection_score:.4f}"
        )

        print(
            f"Epoch time: "
            f"{elapsed:.2f}s"
        )

        history.append(
            {
                "epoch": epoch,
                "train": train_result,
                "validation":
                    validation_result,
                "selection_score":
                    float(
                        selection_score
                    ),
                "epoch_seconds":
                    float(elapsed),
            }
        )

        if (
            selection_score
            > best_score
        ):

            best_score = (
                selection_score
            )

            best_epoch = epoch
            epochs_without_improvement = 0

            checkpoint = {
                "model_state_dict":
                    model.state_dict(),

                "config": {
                    "input_size": 9,
                    "hidden_size":
                        args.hidden_size,

                    "labels":
                        list(LABELS),

                    "relations":
                        list(RELATIONS),

                    "architecture":
                        "Hierarchical Temporal "
                        "Risk-Aware Relational "
                        "GraphSAGE + GRU V5",

                    "loss":
                        "binary focal multi-task",

                    "deadlock_threshold":
                        validation_result[
                            "deadlock_threshold"
                        ],

                    "pre_threshold":
                        validation_result[
                            "pre_threshold"
                        ],

                    "safe_precision_constraint":
                        0.80,

                    "deadlock_precision_constraint":
                        0.98,

                    "selection_score":
                        float(
                            selection_score
                        ),
                },

                "epoch": epoch,

                "best_validation_score":
                    float(best_score),

                "validation_metrics":
                    metrics,

                "validation_risk_metrics":
                    validation_result[
                        "risk_metrics"
                    ],

                "args":
                    vars(args),
            }

            torch.save(
                checkpoint,
                args.output
                / "best_model_v5.pt",
            )

            print()
            print(
                "*** NEW BEST V5 MODEL SAVED ***"
            )

        else:

            epochs_without_improvement += 1

            print(
                f"No improvement for "
                f"{epochs_without_improvement}/"
                f"{args.patience} epoch(s)."
            )

        if (
            epochs_without_improvement
            >= args.patience
        ):

            print(
                "Early stopping."
            )

            break

    with (
        args.output
        / "training_history_v5.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as handle:

        json.dump(
            history,
            handle,
            indent=2,
        )

    print()
    print("=" * 70)
    print("V5 TRAINING COMPLETE")
    print("=" * 70)

    print(
        f"Best epoch: {best_epoch}"
    )

    print(
        f"Best selection score: "
        f"{best_score:.4f}"
    )

    print(
        "Best model:"
    )

    print(
        args.output
        / "best_model_v5.pt"
    )

    print()
    print(
        "TEST SET WAS NOT USED."
    )

    print(
        "Run test_gnn_v5.py next."
    )


if __name__ == "__main__":
    main()
