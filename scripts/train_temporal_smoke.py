#!/usr/bin/env python3
"""Train a small relational GraphSAGE-GRU model to prove dataset trainability."""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from torch import nn


LABELS = {"safe": 0, "pre_deadlock": 1, "deadlocked": 2}
RELATIONS = ("owned_by", "waits_for")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def node_features(node: dict[str, Any]) -> list[float]:
    features = node["features"]
    is_thread = node["type"] == "thread"
    last_cpu = float(features.get("last_cpu", -1))
    return [
        float(is_thread),
        float(not is_thread),
        float(features.get("is_waiting", 0)),
        math.log1p(float(features.get("wait_ns", 0))) / 22.0,
        float(features.get("has_owner", 0)),
        math.log1p(float(features.get("scheduler_switches", 0))) / 8.0,
        math.log1p(float(features.get("wakeups", 0))) / 8.0,
        math.log1p(float(features.get("cpu_migrations", 0))) / 6.0,
        (last_cpu + 1.0) / 8.0,
    ]


def tensorize_graph(snapshot: dict[str, Any]) -> dict[str, torch.Tensor]:
    nodes = snapshot["nodes"]
    node_index = {node["id"]: index for index, node in enumerate(nodes)}
    edges: dict[str, list[list[int]]] = {relation: [[], []] for relation in RELATIONS}
    for edge in snapshot["edges"]:
        relation = edge["type"]
        edges[relation][0].append(node_index[edge["source"]])
        edges[relation][1].append(node_index[edge["target"]])
    return {
        "x": torch.tensor([node_features(node) for node in nodes], dtype=torch.float32),
        "node_type": torch.tensor([0 if node["type"] == "thread" else 1 for node in nodes]),
        **{
            relation: torch.tensor(indices, dtype=torch.long)
            if indices[0] else torch.empty((2, 0), dtype=torch.long)
            for relation, indices in edges.items()
        },
    }


class TemporalSequenceDataset:
    def __init__(self, dataset_dir: Path, split: str):
        snapshots = read_jsonl(dataset_dir / f"{split}.jsonl")
        self.graphs = {
            snapshot["snapshot_id"]: tensorize_graph(snapshot) for snapshot in snapshots
        }
        self.sequences = read_jsonl(dataset_dir / f"{split}_sequences.jsonl")

    def balanced_indices(self, limit: int, seed: int) -> list[int]:
        by_label: dict[str, list[int]] = defaultdict(list)
        for index, sequence in enumerate(self.sequences):
            by_label[sequence["label"]].append(index)
        randomizer = random.Random(seed)
        target = min(min(len(values) for values in by_label.values()), max(1, limit // 3))
        selected = []
        for label in sorted(by_label):
            selected.extend(randomizer.sample(by_label[label], target))
        randomizer.shuffle(selected)
        return selected

    def item(self, index: int) -> tuple[list[dict[str, torch.Tensor]], torch.Tensor]:
        sequence = self.sequences[index]
        graphs = [self.graphs[snapshot_id] for snapshot_id in sequence["snapshot_ids"]]
        return graphs, torch.tensor(LABELS[sequence["label"]], dtype=torch.long)

    def label_counts(self, indices: list[int]) -> dict[str, int]:
        counts = {label: 0 for label in LABELS}
        for index in indices:
            counts[self.sequences[index]["label"]] += 1
        return counts


class RelationalGraphSAGE(nn.Module):
    def __init__(self, input_size: int = 9, hidden_size: int = 32):
        super().__init__()
        self.input = nn.Linear(input_size, hidden_size)
        self.self_linear = nn.Linear(hidden_size, hidden_size)
        self.relation_linears = nn.ModuleDict({
            f"{relation}_{direction}": nn.Linear(hidden_size, hidden_size, bias=False)
            for relation in RELATIONS for direction in ("in", "out")
        })
        self.norm = nn.LayerNorm(hidden_size)

    @staticmethod
    def aggregate(h: torch.Tensor, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        output = torch.zeros_like(h)
        degree = torch.zeros((h.shape[0], 1), dtype=h.dtype, device=h.device)
        if source.numel():
            output.index_add_(0, target, h[source])
            degree.index_add_(0, target, torch.ones((source.shape[0], 1), device=h.device))
        return output / degree.clamp_min(1.0)

    def forward(self, graph: dict[str, torch.Tensor]) -> torch.Tensor:
        h = torch.relu(self.input(graph["x"]))
        updated = self.self_linear(h)
        for relation in RELATIONS:
            source, target = graph[relation]
            incoming = self.aggregate(h, source, target)
            outgoing = self.aggregate(h, target, source)
            updated = updated + self.relation_linears[f"{relation}_in"](incoming)
            updated = updated + self.relation_linears[f"{relation}_out"](outgoing)
        h = torch.relu(self.norm(updated))
        node_type = graph["node_type"]
        pooled = []
        for type_id in (0, 1):
            mask = node_type == type_id
            pooled.append(h[mask].mean(dim=0) if mask.any() else torch.zeros(h.shape[1]))
        return torch.cat(pooled)


class TemporalGraphClassifier(nn.Module):
    def __init__(self, hidden_size: int = 32):
        super().__init__()
        self.graph_encoder = RelationalGraphSAGE(hidden_size=hidden_size)
        self.temporal = nn.GRU(hidden_size * 2, hidden_size, batch_first=True)
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_size), nn.Linear(hidden_size, len(LABELS))
        )

    def forward(self, sequences: list[list[dict[str, torch.Tensor]]]) -> torch.Tensor:
        batch = []
        for sequence in sequences:
            batch.append(torch.stack([self.graph_encoder(graph) for graph in sequence]))
        encoded = torch.stack(batch)
        _, hidden = self.temporal(encoded)
        return self.classifier(hidden[-1])


def batches(indices: list[int], size: int):
    for start in range(0, len(indices), size):
        yield indices[start:start + size]


def evaluate(
    model: nn.Module, dataset: TemporalSequenceDataset, indices: list[int], batch_size: int,
) -> dict[str, Any]:
    model.eval()
    confusion = [[0] * len(LABELS) for _ in LABELS]
    losses = []
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for batch_indices in batches(indices, batch_size):
            items = [dataset.item(index) for index in batch_indices]
            logits = model([item[0] for item in items])
            targets = torch.stack([item[1] for item in items])
            losses.append(float(criterion(logits, targets)))
            predictions = logits.argmax(dim=1)
            for target, prediction in zip(targets.tolist(), predictions.tolist()):
                confusion[target][prediction] += 1
    total = sum(sum(row) for row in confusion)
    correct = sum(confusion[index][index] for index in range(len(LABELS)))
    f1_scores = []
    for index in range(len(LABELS)):
        true_positive = confusion[index][index]
        false_positive = sum(confusion[row][index] for row in range(len(LABELS)) if row != index)
        false_negative = sum(confusion[index][column] for column in range(len(LABELS)) if column != index)
        denominator = 2 * true_positive + false_positive + false_negative
        f1_scores.append(2 * true_positive / denominator if denominator else 0.0)
    return {
        "loss": sum(losses) / len(losses),
        "accuracy": correct / total,
        "macro_f1": sum(f1_scores) / len(f1_scores),
        "confusion_matrix": confusion,
        "samples": total,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--train-samples", type=int, default=3000)
    parser.add_argument("--validation-samples", type=int, default=1200)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260815)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))
    train = TemporalSequenceDataset(args.dataset, "train")
    validation = TemporalSequenceDataset(args.dataset, "validation")
    train_indices = train.balanced_indices(args.train_samples, args.seed)
    validation_indices = validation.balanced_indices(args.validation_samples, args.seed)

    model = TemporalGraphClassifier()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    epoch_losses = []
    gradient_norms = []
    randomizer = random.Random(args.seed)
    for epoch in range(args.epochs):
        model.train()
        randomizer.shuffle(train_indices)
        losses = []
        for batch_indices in batches(train_indices, args.batch_size):
            items = [train.item(index) for index in batch_indices]
            optimizer.zero_grad(set_to_none=True)
            logits = model([item[0] for item in items])
            targets = torch.stack([item[1] for item in items])
            loss = criterion(logits, targets)
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite training loss")
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            if not torch.isfinite(gradient_norm):
                raise RuntimeError("non-finite gradient norm")
            gradient_norms.append(float(gradient_norm))
            optimizer.step()
            losses.append(float(loss.detach()))
        epoch_loss = sum(losses) / len(losses)
        epoch_losses.append(epoch_loss)
        print(f"epoch={epoch + 1} loss={epoch_loss:.6f}", flush=True)

    evaluation = evaluate(model, validation, validation_indices, args.batch_size)
    train_label_counts = train.label_counts(train_indices)
    validation_label_counts = validation.label_counts(validation_indices)
    majority_baseline = max(validation_label_counts.values()) / len(validation_indices)
    report = {
        "model": "relational GraphSAGE + GRU",
        "seed": args.seed,
        "epochs": args.epochs,
        "train_samples": len(train_indices),
        "train_label_counts": train_label_counts,
        "validation": evaluation,
        "validation_label_counts": validation_label_counts,
        "validation_majority_baseline_accuracy": majority_baseline,
        "epoch_losses": epoch_losses,
        "loss_decreased": epoch_losses[-1] < epoch_losses[0],
        "finite_gradients": all(math.isfinite(value) for value in gradient_norms),
        "input_exclusions": [
            "label", "has_cycle", "cycle_nodes", "label_metadata", "provenance",
            "run_id", "snapshot_id", "graph_features",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    torch.save({"model_state_dict": model.state_dict(), "report": report}, args.output.with_suffix(".pt"))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["loss_decreased"] and report["finite_gradients"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
