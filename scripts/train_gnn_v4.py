#!/usr/bin/env python3
"""
Temporal Risk-Aware Relational GraphSAGE + GRU
Deadlock Prediction V4

V4 is designed specifically to improve PRE_DEADLOCK detection.

Main ideas:
1. Relational GraphSAGE encodes each thread/mutex graph.
2. GRU learns temporal evolution across 8 snapshots.
3. Main head predicts:
       SAFE / PRE_DEADLOCK / DEADLOCKED
4. Auxiliary heads predict future deadlock risk:
       within 50 ms
       within 100 ms
       within 300 ms
5. An auxiliary time-to-deadlock signal is learned from the
   future-risk targets when available.
6. PRE_DEADLOCK receives extra classification weight.
7. No SAFE -> PRE_DEADLOCK penalty is used.
8. Validation automatically searches for a PRE_DEADLOCK probability
   threshold instead of blindly using argmax.
9. Best checkpoint is selected using PRE_DEADLOCK F1 with a small
   overall-performance safeguard.

The V3 dataset itself does not need to be regenerated.
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
RELATION_TO_ID = {relation: i for i, relation in enumerate(RELATIONS)}

INPUT_SIZE = 9


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
# JSONL / GZIP loading
# ---------------------------------------------------------------------

def resolve_jsonl_path(dataset_dir: Path, filename: str) -> Path:
    """
    Prefer gzip-compressed JSONL if available.
    Otherwise use normal JSONL.
    """
    gz_path = dataset_dir / f"{filename}.gz"
    jsonl_path = dataset_dir / filename

    if gz_path.exists():
        return gz_path

    if jsonl_path.exists():
        return jsonl_path

    raise FileNotFoundError(
        f"Could not find either:\n"
        f"  {gz_path}\n"
        f"  {jsonl_path}"
    )


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


# ---------------------------------------------------------------------
# Graph tensorization
# ---------------------------------------------------------------------

def node_features(node: dict[str, Any]) -> list[float]:
    """
    Original V3 9-D feature contract.

    0: is_thread
    1: is_lock
    2: is_waiting
    3: normalized log wait_ns
    4: has_owner
    5: normalized scheduler switches
    6: normalized wakeups
    7: normalized CPU migrations
    8: normalized last CPU
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
    """
    Each item contains:

        graphs
        classification label
        future-risk targets
        regression target

    The auxiliary targets come from the existing V3 sequence JSON.
    """

    def __init__(
        self,
        dataset_dir: Path,
        split: str,
    ):
        self.dataset_dir = dataset_dir
        self.split = split

        snapshot_path = resolve_jsonl_path(
            dataset_dir,
            f"{split}.jsonl",
        )

        sequence_path = resolve_jsonl_path(
            dataset_dir,
            f"{split}_sequences.jsonl",
        )

        print(f"Loading {split} snapshots...")
        snapshot_rows = load_jsonl(snapshot_path)

        print(f"Loading {split} sequences...")
        sequence_rows = load_jsonl(sequence_path)

        self.graphs = {
            row["snapshot_id"]: tensorize_graph(row)
            for row in snapshot_rows
        }

        self.sequences: list[dict[str, Any]] = []

        for row in sequence_rows:
            graphs = [
                self.graphs[sid]
                for sid in row["snapshot_ids"]
            ]

            label = LABEL_TO_ID[row["label"]]

            targets = row.get("targets", {})

            within_50 = float(
                bool(targets.get("deadlock_within_50ms", False))
            )

            within_100 = float(
                bool(targets.get("deadlock_within_100ms", False))
            )

            within_300 = float(
                bool(targets.get("deadlock_within_300ms", False))
            )

            # A continuous risk target derived from the existing
            # future-deadlock labels.
            #
            # 0.0 = no deadlock within 300 ms
            # 0.33 = deadlock within 300 ms
            # 0.66 = deadlock within 100 ms
            # 1.0 = deadlock within 50 ms
            #
            # This does NOT create new information outside the dataset.
            if within_50:
                risk_score = 1.0
            elif within_100:
                risk_score = 0.66
            elif within_300:
                risk_score = 0.33
            else:
                risk_score = 0.0

            self.sequences.append(
                {
                    "graphs": graphs,
                    "label": label,
                    "within_50": within_50,
                    "within_100": within_100,
                    "within_300": within_300,
                    "risk_score": risk_score,
                }
            )

        self.labels = [
            item["label"]
            for item in self.sequences
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
    ) -> dict[str, Any]:
        return self.sequences[index]

    def label_counts(self) -> Counter:
        return Counter(self.labels)


# ---------------------------------------------------------------------
# Relational GraphSAGE
# ---------------------------------------------------------------------

class RelationalGraphSAGE(nn.Module):

    def __init__(
        self,
        input_size: int = INPUT_SIZE,
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

        self.norm = nn.LayerNorm(hidden_size)

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

        messages = transform(h[source])

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

        counts = counts.clamp_min(1.0).unsqueeze(1)

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
            thread_pool = h[thread_mask].mean(dim=0)
        else:
            thread_pool = h.mean(dim=0)

        if lock_mask.any():
            lock_pool = h[lock_mask].mean(dim=0)
        else:
            lock_pool = h.mean(dim=0)

        return torch.cat(
            [thread_pool, lock_pool],
            dim=0,
        )


# ---------------------------------------------------------------------
# V4 temporal multi-task model
# ---------------------------------------------------------------------

class TemporalRiskAwareGraphClassifier(nn.Module):

    def __init__(
        self,
        input_size: int = INPUT_SIZE,
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

        self.shared_norm = nn.LayerNorm(
            hidden_size
        )

        # Main classification head.
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, 3),
        )

        # Future-deadlock heads.
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

        # Continuous risk / temporal proximity head.
        self.risk_score = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, 1),
        )

    def forward(
        self,
        batch_graph_sequences: list[list[dict[str, Any]]],
        device: torch.device,
    ) -> dict[str, torch.Tensor]:

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

        temporal = self.shared_norm(
            output[:, -1, :]
        )

        logits = self.classifier(
            temporal
        )

        return {
            "logits": logits,
            "risk_50": self.risk_50(
                temporal
            ).squeeze(1),
            "risk_100": self.risk_100(
                temporal
            ).squeeze(1),
            "risk_300": self.risk_300(
                temporal
            ).squeeze(1),
            "risk_score": self.risk_score(
                temporal
            ).squeeze(1),
        }


# ---------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------

def calculate_metrics(
    predictions: list[int],
    targets: list[int],
) -> dict[str, Any]:

    confusion = np.zeros(
        (3, 3),
        dtype=np.int64,
    )

    for target, prediction in zip(
        targets,
        predictions,
    ):
        confusion[target, prediction] += 1

    per_class = {}

    precisions = []
    recalls = []
    f1s = []

    for class_id, label in enumerate(LABELS):

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

        precision = tp / max(
            1,
            tp + fp,
        )

        recall = tp / max(
            1,
            tp + fn,
        )

        f1 = (
            2 * precision * recall
            / max(
                1e-12,
                precision + recall,
            )
        )

        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

        per_class[label] = {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "support": int(
                confusion[
                    class_id, :
                ].sum()
            ),
        }

    accuracy = (
        np.trace(confusion)
        / max(1, confusion.sum())
    )

    return {
        "accuracy": float(accuracy),
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
        "confusion_matrix": confusion.tolist(),
    }


# ---------------------------------------------------------------------
# PRE threshold
# ---------------------------------------------------------------------

def apply_pre_threshold(
    probabilities: np.ndarray,
    pre_threshold: float,
) -> np.ndarray:
    """
    Normal argmax tends to be conservative about PRE_DEADLOCK.

    We therefore allow PRE_DEADLOCK to win against SAFE when:

        P(PRE) >= threshold

    but DEADLOCKED remains protected by requiring its probability
    to remain the largest probability.

    This only modifies SAFE/PRE decisions.
    """

    predictions = probabilities.argmax(
        axis=1
    )

    safe_id = LABEL_TO_ID["safe"]
    pre_id = LABEL_TO_ID["pre_deadlock"]
    dead_id = LABEL_TO_ID["deadlocked"]

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

        # Never override a strong deadlocked prediction.
        if (
            p_dead >= p_safe
            and p_dead >= p_pre
        ):
            predictions[i] = dead_id

        elif (
            p_pre >= pre_threshold
            and p_pre > p_safe
        ):
            predictions[i] = pre_id

        else:
            predictions[i] = safe_id

    return predictions


def tune_pre_threshold(
    probabilities: np.ndarray,
    targets: list[int],
) -> tuple[float, dict[str, Any]]:

    best_threshold = 0.50
    best_metrics = None
    best_score = -1.0

    # Search a broad range.
    thresholds = np.arange(
        0.10,
        0.71,
        0.02,
    )

    for threshold in thresholds:

        predictions = apply_pre_threshold(
            probabilities,
            float(threshold),
        )

        metrics = calculate_metrics(
            predictions.tolist(),
            targets,
        )

        pre = metrics[
            "per_class"
        ]["pre_deadlock"]

        dead = metrics[
            "per_class"
        ]["deadlocked"]

        # Primary objective:
        # PRE F1.
        #
        # Secondary:
        # preserve deadlock F1.
        #
        # This prevents the threshold from simply
        # predicting PRE for everything.
        score = (
            pre["f1"]
            + 0.20 * dead["f1"]
        )

        if score > best_score:

            best_score = score
            best_threshold = float(
                threshold
            )
            best_metrics = metrics

    return (
        best_threshold,
        best_metrics,
    )


# ---------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------

class WeightedFocalLoss(nn.Module):

    def __init__(
        self,
        class_weights: torch.Tensor,
        gamma: float = 1.0,
    ):
        super().__init__()

        self.register_buffer(
            "class_weights",
            class_weights,
        )

        self.gamma = gamma

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:

        log_probs = F.log_softmax(
            logits,
            dim=1,
        )

        probs = log_probs.exp()

        target_log_probs = log_probs[
            torch.arange(
                targets.size(0),
                device=targets.device,
            ),
            targets,
        ]

        target_probs = probs[
            torch.arange(
                targets.size(0),
                device=targets.device,
            ),
            targets,
        ]

        focal_factor = (
            1.0 - target_probs
        ).pow(self.gamma)

        weights = self.class_weights[
            targets
        ]

        return (
            -weights
            * focal_factor
            * target_log_probs
        ).mean()


# ---------------------------------------------------------------------
# Class weights
# ---------------------------------------------------------------------

def make_class_weights(
    counts: Counter,
    pre_multiplier: float,
) -> torch.Tensor:

    total = sum(
        counts.values()
    )

    weights = []

    for label in LABELS:

        count = max(
            1,
            counts.get(
                LABEL_TO_ID[label],
                0,
            ),
        )

        weight = (
            total
            / (
                len(LABELS)
                * count
            )
        )

        if label == "pre_deadlock":
            weight *= pre_multiplier

        weights.append(weight)

    return torch.tensor(
        weights,
        dtype=torch.float32,
    )


# ---------------------------------------------------------------------
# V4 multi-task loss
# ---------------------------------------------------------------------

def calculate_loss(
    outputs: dict[str, torch.Tensor],
    labels: torch.Tensor,
    within_50: torch.Tensor,
    within_100: torch.Tensor,
    within_300: torch.Tensor,
    risk_score: torch.Tensor,
    classification_criterion: nn.Module,
    risk_weight_50: float,
    risk_weight_100: float,
    risk_weight_300: float,
    regression_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:

    classification_loss = (
        classification_criterion(
            outputs["logits"],
            labels,
        )
    )

    bce = nn.BCEWithLogitsLoss()

    loss_50 = bce(
        outputs["risk_50"],
        within_50,
    )

    loss_100 = bce(
        outputs["risk_100"],
        within_100,
    )

    loss_300 = bce(
        outputs["risk_300"],
        within_300,
    )

    # Regression target is derived from the existing
    # future-risk labels.
    regression_loss = F.smooth_l1_loss(
        torch.sigmoid(
            outputs["risk_score"]
        ),
        risk_score,
    )

    total_loss = (
        classification_loss
        + risk_weight_50 * loss_50
        + risk_weight_100 * loss_100
        + risk_weight_300 * loss_300
        + regression_weight * regression_loss
    )

    return total_loss, {
        "classification": float(
            classification_loss.item()
        ),
        "risk_50": float(
            loss_50.item()
        ),
        "risk_100": float(
            loss_100.item()
        ),
        "risk_300": float(
            loss_300.item()
        ),
        "regression": float(
            regression_loss.item()
        ),
    }


# ---------------------------------------------------------------------
# Epoch
# ---------------------------------------------------------------------

def run_epoch(
    model: nn.Module,
    dataset: TemporalSequenceDataset,
    optimizer: torch.optim.Optimizer | None,
    classification_criterion: nn.Module,
    device: torch.device,
    batch_size: int,
    train: bool,
    seed: int,
    risk_weight_50: float,
    risk_weight_100: float,
    risk_weight_300: float,
    regression_weight: float,
) -> tuple[
    float,
    dict[str, Any],
    np.ndarray,
    list[int],
    dict[str, float],
]:

    model.train(train)

    indices = list(
        range(len(dataset))
    )

    if train:
        random.Random(
            seed
        ).shuffle(indices)

    total_loss = 0.0
    total_samples = 0

    predictions = []
    targets = []

    all_probabilities = []

    loss_components = {
        "classification": 0.0,
        "risk_50": 0.0,
        "risk_100": 0.0,
        "risk_300": 0.0,
        "regression": 0.0,
    }

    for start in range(
        0,
        len(indices),
        batch_size,
    ):

        batch_indices = indices[
            start:start + batch_size
        ]

        batch_items = [
            dataset[i]
            for i in batch_indices
        ]

        batch_graphs = [
            item["graphs"]
            for item in batch_items
        ]

        batch_targets = torch.tensor(
            [
                item["label"]
                for item in batch_items
            ],
            dtype=torch.long,
            device=device,
        )

        batch_50 = torch.tensor(
            [
                item["within_50"]
                for item in batch_items
            ],
            dtype=torch.float32,
            device=device,
        )

        batch_100 = torch.tensor(
            [
                item["within_100"]
                for item in batch_items
            ],
            dtype=torch.float32,
            device=device,
        )

        batch_300 = torch.tensor(
            [
                item["within_300"]
                for item in batch_items
            ],
            dtype=torch.float32,
            device=device,
        )

        batch_risk = torch.tensor(
            [
                item["risk_score"]
                for item in batch_items
            ],
            dtype=torch.float32,
            device=device,
        )

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

            loss, components = calculate_loss(
                outputs,
                batch_targets,
                batch_50,
                batch_100,
                batch_300,
                batch_risk,
                classification_criterion,
                risk_weight_50,
                risk_weight_100,
                risk_weight_300,
                regression_weight,
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

        for key in loss_components:
            loss_components[key] += (
                components[key]
                * batch_count
            )

        probabilities = torch.softmax(
            outputs["logits"],
            dim=1,
        ).detach().cpu().numpy()

        all_probabilities.append(
            probabilities
        )

        predictions.extend(
            outputs["logits"]
            .argmax(dim=1)
            .detach()
            .cpu()
            .tolist()
        )

        targets.extend(
            batch_targets
            .detach()
            .cpu()
            .tolist()
        )

    probabilities = np.concatenate(
        all_probabilities,
        axis=0,
    )

    metrics = calculate_metrics(
        predictions,
        targets,
    )

    average_loss = (
        total_loss
        / max(1, total_samples)
    )

    for key in loss_components:
        loss_components[key] /= max(
            1,
            total_samples,
        )

    return (
        average_loss,
        metrics,
        probabilities,
        targets,
        loss_components,
    )


# ---------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------

def print_metrics(
    prefix: str,
    loss: float,
    metrics: dict[str, Any],
) -> None:

    print(
        f"{prefix} loss:       {loss:.4f}"
    )

    print(
        f"{prefix} accuracy:   "
        f"{metrics['accuracy']:.4f}"
    )

    print(
        f"{prefix} macro-F1:   "
        f"{metrics['macro_f1']:.4f}"
    )

    print(
        f"{prefix} macro-P:    "
        f"{metrics['macro_precision']:.4f}"
    )

    print(
        f"{prefix} macro-R:    "
        f"{metrics['macro_recall']:.4f}"
    )

    for label in LABELS:

        item = metrics[
            "per_class"
        ][label]

        print(
            f"  {label:14s} "
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
        default=Path("outputs/gnn_v4"),
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

    parser.add_argument(
        "--pre-multiplier",
        type=float,
        default=2.5,
        help="Extra weight applied to PRE_DEADLOCK.",
    )

    parser.add_argument(
        "--focal-gamma",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--risk-weight-50",
        type=float,
        default=0.8,
    )

    parser.add_argument(
        "--risk-weight-100",
        type=float,
        default=0.5,
    )

    parser.add_argument(
        "--risk-weight-300",
        type=float,
        default=0.3,
    )

    parser.add_argument(
        "--regression-weight",
        type=float,
        default=0.2,
    )

    parser.add_argument(
        "--limit-train",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--limit-validation",
        type=int,
        default=None,
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
        "TEMPORAL RISK-AWARE "
        "RELATIONAL GRAPHSAGE + GRU"
    )
    print("V4 PRE-DEADLOCK MODEL")
    print("=" * 70)

    print(
        f"Device: {device}"
    )

    print(
        f"PyTorch: {torch.__version__}"
    )

    print(
        f"Dataset: {args.dataset}"
    )

    print(
        f"Output: {args.output}"
    )

    print(
        f"PRE class multiplier: "
        f"{args.pre_multiplier}"
    )

    print(
        f"Focal gamma: "
        f"{args.focal_gamma}"
    )

    print(
        "SAFE -> PRE penalty: DISABLED"
    )

    print()

    # ---------------------------------------------------------------
    # Dataset
    # ---------------------------------------------------------------

    train_dataset = TemporalSequenceDataset(
        args.dataset,
        "train",
    )

    validation_dataset = TemporalSequenceDataset(
        args.dataset,
        "validation",
    )

    if args.limit_train is not None:

        train_dataset.sequences = (
            train_dataset.sequences[
                :args.limit_train
            ]
        )

        train_dataset.labels = [
            item["label"]
            for item
            in train_dataset.sequences
        ]

    if args.limit_validation is not None:

        validation_dataset.sequences = (
            validation_dataset.sequences[
                :args.limit_validation
            ]
        )

        validation_dataset.labels = [
            item["label"]
            for item
            in validation_dataset.sequences
        ]

    train_counts = (
        train_dataset.label_counts()
    )

    validation_counts = (
        validation_dataset.label_counts()
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

    print()
    print("Training class distribution:")

    for label in LABELS:
        print(
            f"  {label:14s}: "
            f"{train_counts.get(LABEL_TO_ID[label], 0):,}"
        )

    print()
    print("Validation class distribution:")

    for label in LABELS:
        print(
            f"  {label:14s}: "
            f"{validation_counts.get(LABEL_TO_ID[label], 0):,}"
        )

    # ---------------------------------------------------------------
    # Class weights
    # ---------------------------------------------------------------

    class_weights = make_class_weights(
        train_counts,
        args.pre_multiplier,
    ).to(device)

    print()
    print("Class weights:")

    for label, weight in zip(
        LABELS,
        class_weights.tolist(),
    ):
        print(
            f"  {label:14s}: "
            f"{weight:.4f}"
        )

    # ---------------------------------------------------------------
    # Model
    # ---------------------------------------------------------------

    model = TemporalRiskAwareGraphClassifier(
        input_size=INPUT_SIZE,
        hidden_size=args.hidden_size,
    ).to(device)

    parameter_count = sum(
        p.numel()
        for p in model.parameters()
    )

    print()
    print(
        f"Model parameters: "
        f"{parameter_count:,}"
    )

    # ---------------------------------------------------------------
    # Optimizer
    # ---------------------------------------------------------------

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    # ---------------------------------------------------------------
    # Classification criterion
    # ---------------------------------------------------------------

    classification_criterion = (
        WeightedFocalLoss(
            class_weights=class_weights,
            gamma=args.focal_gamma,
        )
    )

    args.output.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ---------------------------------------------------------------
    # Training state
    # ---------------------------------------------------------------

    best_score = -1.0
    best_epoch = -1
    best_threshold = 0.50

    epochs_without_improvement = 0

    history = []

    # ---------------------------------------------------------------
    # Training
    # ---------------------------------------------------------------

    print()
    print("=" * 70)
    print("TRAINING")
    print("=" * 70)

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

        (
            train_loss,
            train_metrics,
            _,
            _,
            train_components,
        ) = run_epoch(
            model=model,
            dataset=train_dataset,
            optimizer=optimizer,
            classification_criterion=classification_criterion,
            device=device,
            batch_size=args.batch_size,
            train=True,
            seed=args.seed + epoch,
            risk_weight_50=args.risk_weight_50,
            risk_weight_100=args.risk_weight_100,
            risk_weight_300=args.risk_weight_300,
            regression_weight=args.regression_weight,
        )

        (
            validation_loss,
            validation_argmax_metrics,
            validation_probabilities,
            validation_targets,
            validation_components,
        ) = run_epoch(
            model=model,
            dataset=validation_dataset,
            optimizer=None,
            classification_criterion=classification_criterion,
            device=device,
            batch_size=args.batch_size,
            train=False,
            seed=args.seed,
            risk_weight_50=args.risk_weight_50,
            risk_weight_100=args.risk_weight_100,
            risk_weight_300=args.risk_weight_300,
            regression_weight=args.regression_weight,
        )

        # -----------------------------------------------------------
        # Tune PRE threshold ONLY on validation data.
        # -----------------------------------------------------------

        (
            tuned_threshold,
            tuned_metrics,
        ) = tune_pre_threshold(
            validation_probabilities,
            validation_targets,
        )

        elapsed = (
            time.time()
            - start_time
        )

        print()

        print_metrics(
            "Train",
            train_loss,
            train_metrics,
        )

        print()

        print_metrics(
            "Validation argmax",
            validation_loss,
            validation_argmax_metrics,
        )

        print()

        print(
            "Validation tuned "
            "PRE threshold:"
            f" {tuned_threshold:.2f}"
        )

        print_metrics(
            "Validation tuned",
            validation_loss,
            tuned_metrics,
        )

        print()

        print(
            "Loss components:"
        )

        print(
            f"  classification: "
            f"{train_components['classification']:.4f}"
        )

        print(
            f"  risk 50ms:      "
            f"{train_components['risk_50']:.4f}"
        )

        print(
            f"  risk 100ms:     "
            f"{train_components['risk_100']:.4f}"
        )

        print(
            f"  risk 300ms:     "
            f"{train_components['risk_300']:.4f}"
        )

        print(
            f"  risk score:     "
            f"{train_components['regression']:.4f}"
        )

        print(
            f"Epoch time: "
            f"{elapsed:.2f}s"
        )

        # -----------------------------------------------------------
        # Model selection
        #
        # PRE F1 is the primary target.
        # Deadlock F1 provides a safety constraint.
        # -----------------------------------------------------------

        pre_f1 = tuned_metrics[
            "per_class"
        ]["pre_deadlock"]["f1"]

        dead_f1 = tuned_metrics[
            "per_class"
        ]["deadlocked"]["f1"]

        safe_f1 = tuned_metrics[
            "per_class"
        ]["safe"]["f1"]

        # Weighted objective:
        #
        # PRE matters most.
        # Deadlock must remain strong.
        # Safe performance remains relevant.
        selection_score = (
            0.55 * pre_f1
            + 0.30 * dead_f1
            + 0.15 * safe_f1
        )

        print(
            f"Selection score: "
            f"{selection_score:.4f}"
        )

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_metrics": train_metrics,
                "validation_loss": validation_loss,
                "validation_argmax_metrics":
                    validation_argmax_metrics,
                "validation_tuned_metrics":
                    tuned_metrics,
                "pre_threshold":
                    tuned_threshold,
                "selection_score":
                    selection_score,
                "loss_components":
                    train_components,
                "epoch_seconds":
                    elapsed,
            }
        )

        if selection_score > best_score:

            best_score = selection_score
            best_epoch = epoch
            best_threshold = tuned_threshold
            epochs_without_improvement = 0

            checkpoint = {
                "model_state_dict":
                    model.state_dict(),

                "config": {
                    "input_size":
                        INPUT_SIZE,

                    "hidden_size":
                        args.hidden_size,

                    "labels":
                        list(LABELS),

                    "relations":
                        list(RELATIONS),

                    "architecture":
                        "Temporal Risk-Aware "
                        "Relational GraphSAGE + GRU",

                    "loss":
                        "weighted focal + "
                        "future-risk multitask",

                    "focal_gamma":
                        args.focal_gamma,

                    "pre_multiplier":
                        args.pre_multiplier,

                    "pre_fp_penalty":
                        0.0,

                    "risk_weight_50":
                        args.risk_weight_50,

                    "risk_weight_100":
                        args.risk_weight_100,

                    "risk_weight_300":
                        args.risk_weight_300,

                    "regression_weight":
                        args.regression_weight,

                    "pre_threshold":
                        best_threshold,
                },

                "epoch":
                    epoch,

                "best_selection_score":
                    best_score,

                "best_validation_pre_f1":
                    pre_f1,

                "best_validation_deadlock_f1":
                    dead_f1,

                "best_validation_safe_f1":
                    safe_f1,

                "validation_metrics":
                    tuned_metrics,

                "args":
                    vars(args),
            }

            torch.save(
                checkpoint,
                args.output
                / "best_model_v4.pt",
            )

            print()
            print(
                "*** NEW BEST V4 MODEL SAVED ***"
            )

            print(
                f"Selection score: "
                f"{best_score:.4f}"
            )

            print(
                f"PRE F1: "
                f"{pre_f1:.4f}"
            )

            print(
                f"PRE threshold: "
                f"{best_threshold:.2f}"
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

            print()
            print(
                "Early stopping."
            )

            break

    # ---------------------------------------------------------------
    # Save history
    # ---------------------------------------------------------------

    with (
        args.output
        / "training_history_v4.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as handle:

        json.dump(
            history,
            handle,
            indent=2,
        )

    # ---------------------------------------------------------------
    # Final
    # ---------------------------------------------------------------

    print()
    print("=" * 70)
    print("V4 TRAINING COMPLETE")
    print("=" * 70)

    print(
        f"Best epoch: "
        f"{best_epoch}"
    )

    print(
        f"Best selection score: "
        f"{best_score:.4f}"
    )

    print(
        f"Best PRE threshold: "
        f"{best_threshold:.2f}"
    )

    print(
        "Best model:"
    )

    print(
        args.output
        / "best_model_v4.pt"
    )

    print()
    print(
        "Test set was NOT used."
    )

    print(
        "Run test_gnn_v4.py after training."
    )


if __name__ == "__main__":
    main()
