"""
Train a temporal relational GraphSAGE + GRU model on dataset/v2.

Works on both CPU and CUDA GPU.

Dataset:
    dataset/v2/
        train.jsonl
        train_sequences.jsonl
        validation.jsonl
        validation_sequences.jsonl
        test.jsonl
        test_sequences.jsonl

Classes:
    0 = safe
    1 = pre_deadlock
    2 = deadlocked
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


LABELS = {
    "safe": 0,
    "pre_deadlock": 1,
    "deadlocked": 2,
}

RELATIONS = (
    "owned_by",
    "waits_for",
)


# ============================================================
# Utilities
# ============================================================

def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()

            if line:
                records.append(json.loads(line))

    return records


# ============================================================
# Node features
# ============================================================

def node_features(node: dict[str, Any]) -> list[float]:
    """
    Convert a dataset node into the 9-dimensional feature vector
    used by the repository's GraphSAGE-GRU implementation.

    Features:
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

    features = node["features"]

    is_thread = 1.0 if node["type"] == "thread" else 0.0
    is_lock = 1.0 if node["type"] == "lock" else 0.0

    last_cpu = float(features.get("last_cpu", -1))

    # Keep the same basic treatment as the repository smoke test.
    # -1 means "unknown CPU".
    if last_cpu < 0:
        normalized_cpu = -1.0
    else:
        # CPU IDs are generally small integers. The exact upper bound
        # isn't semantically important here; clipping prevents outliers.
        normalized_cpu = min(last_cpu / 128.0, 1.0)

    return [
        is_thread,
        is_lock,
        float(features.get("is_waiting", 0)),
        math.log1p(float(features.get("wait_ns", 0))) / 22.0,
        float(features.get("has_owner", 0)),
        math.log1p(float(features.get("scheduler_switches", 0))) / 8.0,
        math.log1p(float(features.get("wakeups", 0))) / 8.0,
        math.log1p(float(features.get("cpu_migrations", 0))) / 6.0,
        normalized_cpu,
    ]


# ============================================================
# Graph tensorization
# ============================================================

def tensorize_graph(
    snapshot: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:

    nodes = snapshot["nodes"]

    node_index = {
        node["id"]: index
        for index, node in enumerate(nodes)
    }

    x = torch.tensor(
        [node_features(node) for node in nodes],
        dtype=torch.float32,
        device=device,
    )

    edges: dict[str, list[list[int]]] = {
        relation: [[], []]
        for relation in RELATIONS
    }

    for edge in snapshot["edges"]:
        relation = edge["type"]

        if relation not in edges:
            continue

        source = edge["source"]
        target = edge["target"]

        if source not in node_index or target not in node_index:
            continue

        edges[relation][0].append(node_index[source])
        edges[relation][1].append(node_index[target])

    edge_tensors = {}

    for relation in RELATIONS:
        src, dst = edges[relation]

        if len(src) == 0:
            edge_tensors[relation] = (
                torch.empty((0,), dtype=torch.long, device=device),
                torch.empty((0,), dtype=torch.long, device=device),
            )
        else:
            edge_tensors[relation] = (
                torch.tensor(src, dtype=torch.long, device=device),
                torch.tensor(dst, dtype=torch.long, device=device),
            )

    return {
        "x": x,
        "edges": edge_tensors,
    }


# ============================================================
# Dataset
# ============================================================

class TemporalSequenceDataset:

    def __init__(
        self,
        dataset_dir: Path,
        split: str,
        device: torch.device,
    ):
        self.dataset_dir = dataset_dir
        self.split = split
        self.device = device

        print(f"Loading {split} snapshots...")

        snapshots = read_jsonl(
            dataset_dir / f"{split}.jsonl"
        )

        print(
            f"  Loaded {len(snapshots):,} snapshots"
        )

        # Snapshot ID → tensorized graph
        self.snapshots = {}

        for snapshot in snapshots:
            snapshot_id = snapshot["snapshot_id"]

            self.snapshots[snapshot_id] = tensorize_graph(
                snapshot,
                device,
            )

        print(
            f"  Tensorized {len(self.snapshots):,} snapshots"
        )

        print(f"Loading {split} sequences...")

        self.sequences = read_jsonl(
            dataset_dir / f"{split}_sequences.jsonl"
        )

        print(
            f"  Loaded {len(self.sequences):,} sequences"
        )

    def __len__(self) -> int:
        return len(self.sequences)

    def item(
        self,
        index: int,
    ) -> tuple[list[dict[str, Any]], torch.Tensor]:

        sequence = self.sequences[index]

        graphs = []

        for snapshot_id in sequence["snapshot_ids"]:

            if snapshot_id not in self.snapshots:
                raise KeyError(
                    f"Snapshot {snapshot_id} not found "
                    f"in {self.split}.jsonl"
                )

            graphs.append(
                self.snapshots[snapshot_id]
            )

        label = torch.tensor(
            LABELS[sequence["label"]],
            dtype=torch.long,
            device=self.device,
        )

        return graphs, label

    def label_counts(
        self,
    ) -> dict[str, int]:

        counts = {
            label: 0
            for label in LABELS
        }

        for sequence in self.sequences:
            counts[sequence["label"]] += 1

        return counts


# ============================================================
# Relational GraphSAGE
# ============================================================

class RelationalGraphSAGE(nn.Module):

    def __init__(
        self,
        input_size: int = 9,
        hidden_size: int = 32,
    ):
        super().__init__()

        self.hidden_size = hidden_size

        self.input_projection = nn.Linear(
            input_size,
            hidden_size,
        )

        self.relation_layers = nn.ModuleDict({
            relation: nn.Linear(
                hidden_size,
                hidden_size,
                bias=False,
            )
            for relation in RELATIONS
        })

        self.self_layer = nn.Linear(
            hidden_size,
            hidden_size,
        )

        self.norm = nn.LayerNorm(
            hidden_size
        )

    @staticmethod
    def aggregate(
        h: torch.Tensor,
        source: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:

        """
        Mean aggregate source-node representations into target nodes.
        """

        num_nodes = h.size(0)
        hidden_size = h.size(1)

        if source.numel() == 0:
            return torch.zeros(
                (num_nodes, hidden_size),
                dtype=h.dtype,
                device=h.device,
            )

        messages = h[source]

        result = torch.zeros(
            (num_nodes, hidden_size),
            dtype=h.dtype,
            device=h.device,
        )

        result.index_add_(
            0,
            target,
            messages,
        )

        degree = torch.zeros(
            num_nodes,
            dtype=h.dtype,
            device=h.device,
        )

        degree.index_add_(
            0,
            target,
            torch.ones_like(target, dtype=h.dtype),
        )

        degree = degree.clamp_min(1.0).unsqueeze(1)

        return result / degree

    def forward(
        self,
        graph: dict[str, Any],
    ) -> torch.Tensor:

        x = graph["x"]

        h = F.relu(
            self.input_projection(x)
        )

        relation_messages = []

        for relation in RELATIONS:

            source, target = graph["edges"][relation]

            aggregated = self.aggregate(
                h,
                source,
                target,
            )

            transformed = self.relation_layers[relation](
                aggregated
            )

            relation_messages.append(
                transformed
            )

        combined = self.self_layer(h)

        for message in relation_messages:
            combined = combined + message

        h = F.relu(
            self.norm(combined)
        )

        # Separate thread and lock graph pooling.
        node_types = graph.get(
            "node_types",
            None,
        )

        # We don't currently store node types in tensorized graphs,
        # so use the first two feature columns.
        is_thread = x[:, 0] > 0.5
        is_lock = x[:, 1] > 0.5

        if is_thread.any():
            thread_embedding = h[is_thread].mean(dim=0)
        else:
            thread_embedding = torch.zeros(
                self.hidden_size,
                device=h.device,
                dtype=h.dtype,
            )

        if is_lock.any():
            lock_embedding = h[is_lock].mean(dim=0)
        else:
            lock_embedding = torch.zeros(
                self.hidden_size,
                device=h.device,
                dtype=h.dtype,
            )

        return torch.cat(
            [
                thread_embedding,
                lock_embedding,
            ],
            dim=0,
        )


# ============================================================
# Temporal model
# ============================================================

class TemporalGraphClassifier(nn.Module):

    def __init__(
        self,
        hidden_size: int = 32,
        num_classes: int = 3,
    ):
        super().__init__()

        self.graph_encoder = RelationalGraphSAGE(
            input_size=9,
            hidden_size=hidden_size,
        )

        self.temporal = nn.GRU(
            input_size=hidden_size * 2,
            hidden_size=hidden_size,
            batch_first=True,
        )

        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(
                hidden_size,
                num_classes,
            ),
        )

    def forward(
        self,
        sequences: list[list[dict[str, Any]]],
    ) -> torch.Tensor:

        sequence_embeddings = []

        for graphs in sequences:

            graph_embeddings = []

            for graph in graphs:
                graph_embeddings.append(
                    self.graph_encoder(graph)
                )

            graph_embeddings = torch.stack(
                graph_embeddings,
                dim=0,
            )

            sequence_embeddings.append(
                graph_embeddings
            )

        x = torch.stack(
            sequence_embeddings,
            dim=0,
        )

        output, _ = self.temporal(x)

        final_state = output[:, -1, :]

        return self.classifier(
            final_state
        )


# ============================================================
# Metrics
# ============================================================

def calculate_metrics(
    targets: list[int],
    predictions: list[int],
) -> dict[str, Any]:

    num_classes = len(LABELS)

    confusion = [
        [0 for _ in range(num_classes)]
        for _ in range(num_classes)
    ]

    for target, prediction in zip(
        targets,
        predictions,
    ):
        confusion[target][prediction] += 1

    total = len(targets)

    correct = sum(
        confusion[i][i]
        for i in range(num_classes)
    )

    accuracy = (
        correct / total
        if total
        else 0.0
    )

    precision_values = []
    recall_values = []
    f1_values = []

    per_class = {}

    inverse_labels = {
        value: key
        for key, value in LABELS.items()
    }

    for i in range(num_classes):

        tp = confusion[i][i]

        fp = sum(
            confusion[row][i]
            for row in range(num_classes)
            if row != i
        )

        fn = sum(
            confusion[i][column]
            for column in range(num_classes)
            if column != i
        )

        precision = (
            tp / (tp + fp)
            if tp + fp
            else 0.0
        )

        recall = (
            tp / (tp + fn)
            if tp + fn
            else 0.0
        )

        f1 = (
            2 * precision * recall /
            (precision + recall)
            if precision + recall
            else 0.0
        )

        precision_values.append(precision)
        recall_values.append(recall)
        f1_values.append(f1)

        per_class[inverse_labels[i]] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": tp + fn,
        }

    macro_precision = sum(
        precision_values
    ) / num_classes

    macro_recall = sum(
        recall_values
    ) / num_classes

    macro_f1 = sum(
        f1_values
    ) / num_classes

    return {
        "accuracy": accuracy,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "per_class": per_class,
        "confusion_matrix": confusion,
    }


# ============================================================
# Class weights
# ============================================================

def make_class_weights(
    counts: dict[str, int],
    device: torch.device,
) -> torch.Tensor:

    total = sum(counts.values())
    num_classes = len(counts)

    weights = []

    for label in (
        "safe",
        "pre_deadlock",
        "deadlocked",
    ):

        count = counts[label]

        weight = (
            total /
            (num_classes * count)
        )

        weights.append(weight)

    return torch.tensor(
        weights,
        dtype=torch.float32,
        device=device,
    )


# ============================================================
# Training / evaluation
# ============================================================

def run_epoch(
    model: nn.Module,
    dataset: TemporalSequenceDataset,
    indices: list[int],
    optimizer: torch.optim.Optimizer | None,
    criterion: nn.Module,
    batch_size: int,
    train: bool,
) -> tuple[float, dict[str, Any]]:

    model.train(train)

    random_indices = indices.copy()

    if train:
        random.shuffle(random_indices)

    total_loss = 0.0

    targets = []
    predictions = []

    num_batches = math.ceil(
        len(random_indices) / batch_size
    )

    for batch_number in range(num_batches):

        start = batch_number * batch_size
        end = min(
            start + batch_size,
            len(random_indices),
        )

        batch_indices = random_indices[
            start:end
        ]

        items = [
            dataset.item(index)
            for index in batch_indices
        ]

        sequences = [
            item[0]
            for item in items
        ]

        labels = torch.stack(
            [
                item[1]
                for item in items
            ]
        )

        if train:
            optimizer.zero_grad(
                set_to_none=True
            )

        logits = model(
            sequences
        )

        loss = criterion(
            logits,
            labels,
        )

        if train:

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0,
            )

            optimizer.step()

        total_loss += (
            loss.item() *
            len(batch_indices)
        )

        predicted = logits.argmax(
            dim=1
        )

        targets.extend(
            labels.detach()
            .cpu()
            .tolist()
        )

        predictions.extend(
            predicted.detach()
            .cpu()
            .tolist()
        )

        if (
            train
            and (
                batch_number == 0
                or (batch_number + 1) % 25 == 0
                or batch_number + 1 == num_batches
            )
        ):
            print(
                f"\r  Batch "
                f"{batch_number + 1}/{num_batches}",
                end="",
                flush=True,
            )

    if train:
        print()

    average_loss = (
        total_loss / len(random_indices)
    )

    metrics = calculate_metrics(
        targets,
        predictions,
    )

    return average_loss, metrics


# ============================================================
# Main
# ============================================================

def main() -> int:

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "outputs/gnn"
        ),
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
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

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 70)
    print("Temporal Relational GraphSAGE + GRU")
    print("=" * 70)

    print(
        f"Device: {device}"
    )

    if device.type == "cuda":
        print(
            f"GPU: "
            f"{torch.cuda.get_device_name(0)}"
        )

    print(
        f"PyTorch: {torch.__version__}"
    )

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    train = TemporalSequenceDataset(
        args.dataset,
        "train",
        device,
    )

    validation = TemporalSequenceDataset(
        args.dataset,
        "validation",
        device,
    )

    train_indices = list(
        range(len(train))
    )

    validation_indices = list(
        range(len(validation))
    )

    if args.limit_train is not None:
        train_indices = train_indices[
            :args.limit_train
        ]

    if args.limit_validation is not None:
        validation_indices = validation_indices[
            :args.limit_validation
        ]

    print()
    print("Training samples:", len(train_indices))
    print("Validation samples:", len(validation_indices))

    print()
    print("Training class distribution:")

    train_counts = train.label_counts()

    for label, count in train_counts.items():
        print(
            f"  {label:15s}: {count:,}"
        )

    # --------------------------------------------------------
    # Class weights
    # --------------------------------------------------------

    class_weights = make_class_weights(
        train_counts,
        device,
    )

    print()
    print("Class weights:")

    for label, weight in zip(
        LABELS,
        class_weights.tolist(),
    ):
        print(
            f"  {label:15s}: {weight:.4f}"
        )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = TemporalGraphClassifier(
        hidden_size=args.hidden_size,
        num_classes=len(LABELS),
    ).to(device)

    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    print()
    print(
        f"Model parameters: "
        f"{parameter_count:,}"
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    criterion = nn.CrossEntropyLoss(
        weight=class_weights
    )

    # --------------------------------------------------------
    # Output directory
    # --------------------------------------------------------

    args.output.mkdir(
        parents=True,
        exist_ok=True,
    )

    best_model_path = (
        args.output /
        "best_model.pt"
    )

    history_path = (
        args.output /
        "history.json"
    )

    history = []

    best_f1 = -1.0
    epochs_without_improvement = 0

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("TRAINING")
    print("=" * 70)

    for epoch in range(
        1,
        args.epochs + 1,
    ):

        print()
        print(
            f"Epoch {epoch}/{args.epochs}"
        )

        start_time = time.time()

        # -----------------------------
        # Training
        # -----------------------------

        train_loss, train_metrics = run_epoch(
            model=model,
            dataset=train,
            indices=train_indices,
            optimizer=optimizer,
            criterion=criterion,
            batch_size=args.batch_size,
            train=True,
        )

        # -----------------------------
        # Validation
        # -----------------------------

        validation_loss, validation_metrics = run_epoch(
            model=model,
            dataset=validation,
            indices=validation_indices,
            optimizer=None,
            criterion=criterion,
            batch_size=args.batch_size,
            train=False,
        )

        elapsed = time.time() - start_time

        print()
        print(
            f"Train loss:       {train_loss:.4f}"
        )

        print(
            f"Validation loss:  {validation_loss:.4f}"
        )

        print(
            f"Train accuracy:    "
            f"{train_metrics['accuracy']:.4f}"
        )

        print(
            f"Validation accuracy: "
            f"{validation_metrics['accuracy']:.4f}"
        )

        print(
            f"Validation macro-F1: "
            f"{validation_metrics['macro_f1']:.4f}"
        )

        print(
            f"Epoch time:        "
            f"{elapsed:.2f}s"
        )

        print()
        print("Validation per-class:")

        for label, values in (
            validation_metrics["per_class"]
            .items()
        ):

            print(
                f"  {label:15s} "
                f"P={values['precision']:.4f} "
                f"R={values['recall']:.4f} "
                f"F1={values['f1']:.4f} "
                f"N={values['support']}"
            )

        epoch_record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "validation_loss": validation_loss,
            "train_metrics": train_metrics,
            "validation_metrics": validation_metrics,
            "epoch_seconds": elapsed,
        }

        history.append(
            epoch_record
        )

        # ----------------------------------------------------
        # Save history
        # ----------------------------------------------------

        history_path.write_text(
            json.dumps(
                history,
                indent=2,
            ),
            encoding="utf-8",
        )

        # ----------------------------------------------------
        # Save best model
        # ----------------------------------------------------

        current_f1 = (
            validation_metrics["macro_f1"]
        )

        if current_f1 > best_f1:

            best_f1 = current_f1
            epochs_without_improvement = 0

            torch.save(
                {
                    "model_state_dict":
                        model.state_dict(),
                    "optimizer_state_dict":
                        optimizer.state_dict(),
                    "epoch": epoch,
                    "validation_macro_f1":
                        current_f1,
                    "label_mapping":
                        LABELS,
                    "hidden_size":
                        args.hidden_size,
                },
                best_model_path,
            )

            print()
            print(
                f"✓ New best model saved "
                f"(macro-F1={best_f1:.4f})"
            )

        else:

            epochs_without_improvement += 1

            print(
                f"No improvement for "
                f"{epochs_without_improvement} "
                f"epoch(s)"
            )

        # ----------------------------------------------------
        # Early stopping
        # ----------------------------------------------------

        if (
            epochs_without_improvement
            >= args.patience
        ):

            print()
            print(
                "Early stopping."
            )

            break

    # --------------------------------------------------------
    # Finish
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)

    print(
        f"Best validation macro-F1: "
        f"{best_f1:.4f}"
    )

    print(
        f"Best model: "
        f"{best_model_path}"
    )

    print(
        f"History: "
        f"{history_path}"
    )

    print()
    print(
        "Test set was NOT used."
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())