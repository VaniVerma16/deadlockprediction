#!/usr/bin/env python3
"""
Train Temporal Relational GraphSAGE + GRU for deadlock prediction.

This version keeps the same 9-D node input and model dimensions as the
original train_gnn.py so the existing test_gnn.py can load the checkpoint.

Changes aimed at V3 pre-deadlock performance:
1. Focal loss focuses learning on hard/misclassified examples.
2. A small explicit penalty discourages SAFE -> PRE_DEADLOCK false positives,
   which were the main V3 test error.
3. Training remains selected by validation Macro-F1.
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


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load plain JSONL or gzip-compressed JSONL (.jsonl.gz)."""
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
    """Prefer .jsonl.gz when present, otherwise fall back to .jsonl."""
    compressed = dataset_dir / f"{filename}.gz"
    plain = dataset_dir / filename

    if compressed.exists():
        return compressed
    if plain.exists():
        return plain

    raise FileNotFoundError(
        f"Could not find either {compressed} or {plain}"
    )


def node_features(node: dict[str, Any]) -> list[float]:
    """
    Keep the original 9-D feature contract.

    0 is_thread
    1 is_lock
    2 is_waiting
    3 log1p(wait_ns) / 22
    4 has_owner
    5 log1p(scheduler_switches) / 8
    6 log1p(wakeups) / 8
    7 log1p(cpu_migrations) / 6
    8 normalized last_cpu
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
    node_index = {node["id"]: i for i, node in enumerate(nodes)}

    x = torch.tensor(
        [node_features(node) for node in nodes],
        dtype=torch.float32,
    )

    edge_lists: dict[str, list[tuple[int, int]]] = {
        relation: [] for relation in RELATIONS
    }

    for edge in snapshot.get("edges", []):
        relation = edge["type"]
        if relation not in edge_lists:
            continue
        source = node_index[edge["source"]]
        target = node_index[edge["target"]]
        edge_lists[relation].append((source, target))

    edges: dict[str, torch.Tensor] = {}
    for relation in RELATIONS:
        pairs = edge_lists[relation]
        if pairs:
            edges[relation] = torch.tensor(
                pairs, dtype=torch.long
            ).t().contiguous()
        else:
            edges[relation] = torch.empty((2, 0), dtype=torch.long)

    return {"x": x, "edges": edges}


class TemporalSequenceDataset:
    def __init__(self, dataset_dir: Path, split: str):
        self.dataset_dir = dataset_dir
        self.split = split

        snapshot_rows = load_jsonl(resolve_jsonl_path(dataset_dir, f"{split}.jsonl"))
        sequence_rows = load_jsonl(resolve_jsonl_path(dataset_dir, f"{split}_sequences.jsonl"))

        self.graphs = {
            row["snapshot_id"]: tensorize_graph(row)
            for row in snapshot_rows
        }

        self.sequences: list[tuple[list[dict[str, Any]], int]] = []
        for row in sequence_rows:
            graphs = [self.graphs[sid] for sid in row["snapshot_ids"]]
            label = LABEL_TO_ID[row["label"]]
            self.sequences.append((graphs, label))

        self.labels = [label for _, label in self.sequences]

        print(f"Loaded {len(snapshot_rows):,} snapshots")
        print(f"Loaded {len(self.sequences):,} sequences")

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, index: int) -> tuple[list[dict[str, Any]], int]:
        return self.sequences[index]

    def label_counts(self) -> Counter:
        return Counter(self.labels)


class RelationalGraphSAGE(nn.Module):
    def __init__(self, input_size: int = 9, hidden_size: int = 32):
        super().__init__()

        self.input_proj = nn.Linear(input_size, hidden_size)
        self.self_layer = nn.Linear(hidden_size, hidden_size)

        self.relation_layers = nn.ModuleDict({
            relation: nn.Linear(hidden_size, hidden_size, bias=False)
            for relation in RELATIONS
        })

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
        aggregated.index_add_(0, target, messages)

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

    def forward(self, graph: dict[str, Any]) -> torch.Tensor:
        x = graph["x"]
        edges = graph["edges"]

        h = F.relu(self.input_proj(x))

        out = self.self_layer(h)

        for relation in RELATIONS:
            out = out + self.aggregate(
                h,
                edges[relation],
                self.relation_layers[relation],
            )

        h = F.relu(self.norm(out))

        # The first two feature columns identify thread/lock nodes.
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

        return torch.cat([thread_pool, lock_pool], dim=0)


class TemporalGraphClassifier(nn.Module):
    def __init__(self, input_size: int = 9, hidden_size: int = 32):
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

        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, 3),
        )

    def forward(
        self,
        batch_graph_sequences: list[list[dict[str, Any]]],
        device: torch.device,
    ) -> torch.Tensor:
        sequence_embeddings: list[torch.Tensor] = []

        for graphs in batch_graph_sequences:
            embeddings = []
            for graph in graphs:
                graph = {
                    "x": graph["x"].to(device),
                    "edges": {
                        relation: edge.to(device)
                        for relation, edge in graph["edges"].items()
                    },
                }
                embeddings.append(self.gnn(graph))

            sequence_embeddings.append(torch.stack(embeddings))

        # All generated V3 sequences use the same sequence length.
        x = torch.stack(sequence_embeddings, dim=0)
        output, _ = self.gru(x)

        return self.classifier(output[:, -1, :])


class FocalCrossEntropy(nn.Module):
    """
    Weighted focal loss.

    gamma=1.5 is deliberately moderate: V3 already detects deadlocks very
    well, so we want more emphasis on difficult SAFE/PRE boundaries without
    completely dominating the loss.
    """

    def __init__(
        self,
        class_weights: torch.Tensor,
        gamma: float = 1.5,
    ):
        super().__init__()
        self.register_buffer("class_weights", class_weights)
        self.gamma = gamma

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        log_probs = F.log_softmax(logits, dim=1)
        probs = log_probs.exp()

        target_log_probs = log_probs[
            torch.arange(targets.size(0), device=targets.device),
            targets,
        ]
        target_probs = probs[
            torch.arange(targets.size(0), device=targets.device),
            targets,
        ]

        focal_factor = (1.0 - target_probs).pow(self.gamma)
        weights = self.class_weights[targets]

        loss = -weights * focal_factor * target_log_probs
        return loss.mean()


def pre_deadlock_false_positive_penalty(
    logits: torch.Tensor,
    targets: torch.Tensor,
    strength: float,
) -> torch.Tensor:
    """
    Penalize predicting PRE_DEADLOCK when the true label is SAFE.

    V3's main test error was SAFE -> PRE_DEADLOCK. This term is small so that
    pre-deadlock recall is not sacrificed for precision.
    """
    if strength <= 0:
        return logits.new_zeros(())

    probabilities = F.softmax(logits, dim=1)
    safe_mask = targets == LABEL_TO_ID["safe"]

    if not safe_mask.any():
        return logits.new_zeros(())

    pre_probability = probabilities[safe_mask, LABEL_TO_ID["pre_deadlock"]]

    return strength * pre_probability.mean()


def make_class_weights(
    counts: Counter,
    pre_multiplier: float = 1.0,
) -> torch.Tensor:
    total = sum(counts.values())
    weights = []

    for label in LABELS:
        count = max(1, counts.get(label, 0))
        weight = total / (len(LABELS) * count)

        if label == "pre_deadlock":
            weight *= pre_multiplier

        weights.append(weight)

    return torch.tensor(weights, dtype=torch.float32)


@torch.no_grad()
def calculate_metrics(
    predictions: list[int],
    targets: list[int],
) -> dict[str, Any]:
    confusion = np.zeros((3, 3), dtype=np.int64)

    for target, prediction in zip(targets, predictions):
        confusion[target, prediction] += 1

    per_class = {}

    precisions = []
    recalls = []
    f1s = []

    for class_id, label in enumerate(LABELS):
        tp = confusion[class_id, class_id]
        fp = confusion[:, class_id].sum() - tp
        fn = confusion[class_id, :].sum() - tp

        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1 = (
            2 * precision * recall / max(1e-12, precision + recall)
        )

        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

        per_class[label] = {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "support": int(confusion[class_id, :].sum()),
        }

    accuracy = np.trace(confusion) / max(1, confusion.sum())

    return {
        "accuracy": float(accuracy),
        "macro_precision": float(np.mean(precisions)),
        "macro_recall": float(np.mean(recalls)),
        "macro_f1": float(np.mean(f1s)),
        "per_class": per_class,
        "confusion_matrix": confusion.tolist(),
    }


def run_epoch(
    model: nn.Module,
    dataset: TemporalSequenceDataset,
    optimizer: torch.optim.Optimizer | None,
    criterion: nn.Module,
    device: torch.device,
    batch_size: int,
    focal_gamma: float,
    pre_fp_penalty: float,
    train: bool,
    seed: int,
) -> tuple[float, dict[str, Any]]:
    model.train(train)

    indices = list(range(len(dataset)))

    if train:
        random.Random(seed).shuffle(indices)

    total_loss = 0.0
    total_samples = 0

    predictions: list[int] = []
    targets: list[int] = []

    for start in range(0, len(indices), batch_size):
        batch_indices = indices[start:start + batch_size]

        batch_graphs = [dataset[i][0] for i in batch_indices]
        batch_targets = torch.tensor(
            [dataset[i][1] for i in batch_indices],
            dtype=torch.long,
            device=device,
        )

        if train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(train):
            logits = model(batch_graphs, device)

            loss = criterion(logits, batch_targets)

            # Additional boundary-aware term for the dominant V3 error.
            loss = loss + pre_deadlock_false_positive_penalty(
                logits,
                batch_targets,
                pre_fp_penalty,
            )

            if train:
                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=1.0,
                )

                optimizer.step()

        batch_count = len(batch_indices)
        total_loss += float(loss.item()) * batch_count
        total_samples += batch_count

        predictions.extend(logits.argmax(dim=1).detach().cpu().tolist())
        targets.extend(batch_targets.detach().cpu().tolist())

    metrics = calculate_metrics(predictions, targets)
    average_loss = total_loss / max(1, total_samples)

    return average_loss, metrics


def print_metrics(
    prefix: str,
    loss: float,
    metrics: dict[str, Any],
) -> None:
    print(
        f"{prefix} loss: {loss:.4f} | "
        f"accuracy: {metrics['accuracy']:.4f} | "
        f"macro-F1: {metrics['macro_f1']:.4f} | "
        f"macro-P: {metrics['macro_precision']:.4f} | "
        f"macro-R: {metrics['macro_recall']:.4f}"
    )

    for label in LABELS:
        item = metrics["per_class"][label]
        print(
            f"  {label:12s} "
            f"P={item['precision']:.4f} "
            f"R={item['recall']:.4f} "
            f"F1={item['f1']:.4f} "
            f"N={item['support']}"
        )


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
        default=Path("outputs/gnn_v3_improved"),
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
        "--focal-gamma",
        type=float,
        default=1.5,
        help="Focal-loss gamma. 0 disables focal focusing.",
    )
    parser.add_argument(
        "--pre-multiplier",
        type=float,
        default=1.0,
        help="Extra multiplier for the pre_deadlock class weight.",
    )
    parser.add_argument(
        "--pre-fp-penalty",
        type=float,
        default=0.15,
        help="Penalty on SAFE samples assigned probability of pre_deadlock.",
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
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("=" * 70)
    print("TRAINING TEMPORAL RELATIONAL GRAPHSAGE + GRU (V3 IMPROVED)")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"PyTorch: {torch.__version__}")

    if torch.cuda.is_available():
        print(f"CUDA: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA: False")
        print("GPU: CPU only")

    print(f"Dataset: {args.dataset}")
    print(f"Output: {args.output}")
    print(f"Focal gamma: {args.focal_gamma}")
    print(f"Pre-deadlock class multiplier: {args.pre_multiplier}")
    print(f"SAFE -> PRE false-positive penalty: {args.pre_fp_penalty}")

    train_dataset = TemporalSequenceDataset(
        args.dataset,
        "train",
    )
    validation_dataset = TemporalSequenceDataset(
        args.dataset,
        "validation",
    )

    if args.limit_train is not None:
        train_dataset.sequences = train_dataset.sequences[:args.limit_train]
        train_dataset.labels = [label for _, label in train_dataset.sequences]

    if args.limit_validation is not None:
        validation_dataset.sequences = (
            validation_dataset.sequences[:args.limit_validation]
        )
        validation_dataset.labels = [
            label for _, label in validation_dataset.sequences
        ]

    print()
    print(f"Training samples: {len(train_dataset):,}")
    print(f"Validation samples: {len(validation_dataset):,}")
    print(f"Training class distribution: {dict(train_dataset.label_counts())}")
    print(
        f"Validation class distribution: "
        f"{dict(validation_dataset.label_counts())}"
    )

    class_weights = make_class_weights(
        train_dataset.label_counts(),
        pre_multiplier=args.pre_multiplier,
    ).to(device)

    print(f"Class weights:")
    for label, weight in zip(LABELS, class_weights.tolist()):
        print(f"  {label}: {weight:.4f}")

    model = TemporalGraphClassifier(
        input_size=9,
        hidden_size=args.hidden_size,
    ).to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    if args.focal_gamma > 0:
        criterion = FocalCrossEntropy(
            class_weights=class_weights,
            gamma=args.focal_gamma,
        )
    else:
        criterion = nn.CrossEntropyLoss(
            weight=class_weights,
        )

    args.output.mkdir(parents=True, exist_ok=True)

    best_macro_f1 = -1.0
    best_epoch = -1
    epochs_without_improvement = 0

    history: list[dict[str, Any]] = []

    for epoch in range(1, args.epochs + 1):
        print()
        print("-" * 70)
        print(f"Epoch {epoch}/{args.epochs}")

        start_time = time.time()

        train_loss, train_metrics = run_epoch(
            model=model,
            dataset=train_dataset,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            batch_size=args.batch_size,
            focal_gamma=args.focal_gamma,
            pre_fp_penalty=args.pre_fp_penalty,
            train=True,
            seed=args.seed + epoch,
        )

        validation_loss, validation_metrics = run_epoch(
            model=model,
            dataset=validation_dataset,
            optimizer=None,
            criterion=criterion,
            device=device,
            batch_size=args.batch_size,
            focal_gamma=args.focal_gamma,
            pre_fp_penalty=args.pre_fp_penalty,
            train=False,
            seed=args.seed,
        )

        elapsed = time.time() - start_time

        print_metrics("Train", train_loss, train_metrics)
        print_metrics("Validation", validation_loss, validation_metrics)
        print(f"Epoch time: {elapsed:.2f}s")

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_metrics": train_metrics,
            "validation_loss": validation_loss,
            "validation_metrics": validation_metrics,
            "epoch_seconds": elapsed,
        })

        current_f1 = validation_metrics["macro_f1"]

        if current_f1 > best_macro_f1:
            best_macro_f1 = current_f1
            best_epoch = epoch
            epochs_without_improvement = 0

            checkpoint = {
                "model_state_dict": model.state_dict(),
                "config": {
                    "input_size": 9,
                    "hidden_size": args.hidden_size,
                    "labels": list(LABELS),
                    "relations": list(RELATIONS),
                    "architecture": "Temporal Relational GraphSAGE + GRU",
                    "loss": "weighted focal cross entropy",
                    "focal_gamma": args.focal_gamma,
                    "pre_multiplier": args.pre_multiplier,
                    "pre_fp_penalty": args.pre_fp_penalty,
                },
                "epoch": epoch,
                "best_validation_macro_f1": best_macro_f1,
                "validation_metrics": validation_metrics,
                "args": vars(args),
            }

            torch.save(
                checkpoint,
                args.output / "best_model.pt",
            )

            print(
                f"*** New best model saved. "
                f"Validation Macro-F1: {best_macro_f1:.4f}"
            )
        else:
            epochs_without_improvement += 1
            print(
                f"No improvement for "
                f"{epochs_without_improvement}/{args.patience} epoch(s)."
            )

        if epochs_without_improvement >= args.patience:
            print("Early stopping.")
            break

    with (args.output / "training_history.json").open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(history, handle, indent=2)

    print()
    print("=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"Best epoch: {best_epoch}")
    print(f"Best validation Macro-F1: {best_macro_f1:.4f}")
    print(f"Model: {args.output / 'best_model.pt'}")
    print()
    print("Test set was NOT used.")
    print("Use test_gnn.py once after selecting the final model.")


if __name__ == "__main__":
    main()
