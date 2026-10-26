#!/usr/bin/env python3
"""
Controlled experiments for the V3 deadlock dataset.

Experiment 1:
  - Train a standard 3-class GraphSAGE + GRU classifier.
  - Cross-entropy + argmax.
  - Same V3 train/validation/test split, 8-snapshot sequences,
    9-D node features, hidden size, optimizer and training budget as V5.

Experiment 2:
  - Compare full V5 multi-task training against the same V5 architecture
    trained without the three future-risk auxiliary losses.
  - Both use the same deadlock/pre-deadlock focal losses and validation-only
    hierarchical threshold selection.
  - This isolates the contribution of future-risk auxiliary supervision.

Run from repository root:
  python scripts/graphsagegrucomp_nofuturerisk_study.py --dataset dataset/v3

This script imports the project's existing V5 dataset/model utilities from
scripts/train_gnn_v5.py. It does NOT change the original model/checkpoint.
"""

from __future__ import annotations

import argparse
import json
import random
import time
import pathlib
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import classification_report, confusion_matrix

# Since this file is in scripts/, running it as shown puts scripts/ on sys.path.
from train_gnn_v5 import (
    BinaryFocalLoss,
    LABELS,
    LABEL_TO_ID,
    RELATIONS,
    RelationalGraphSAGE,
    TemporalRiskAwareV5,
    TemporalSequenceDataset,
    calculate_metrics,
    get_targets,
    hierarchical_predictions,
    set_seed,
    tune_thresholds,
)


class ThreeClassGraphSAGEGRU(nn.Module):
    """Same graph encoder pattern, with a conventional 3-class softmax head."""

    def __init__(self, input_size: int = 9, hidden_size: int = 32):
        super().__init__()
        self.gnn = RelationalGraphSAGE(input_size=input_size, hidden_size=hidden_size)
        self.gru = nn.GRU(
            input_size=hidden_size * 2,
            hidden_size=hidden_size,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(hidden_size)
        self.classifier = nn.Linear(hidden_size, 3)

    def forward(self, graph_sequences, device):
        sequence_embeddings = []
        for graphs in graph_sequences:
            embeddings = []
            for graph in graphs:
                moved = {
                    "x": graph["x"].to(device),
                    "edges": {r: e.to(device) for r, e in graph["edges"].items()},
                }
                embeddings.append(self.gnn(moved))
            sequence_embeddings.append(torch.stack(embeddings))
        x = torch.stack(sequence_embeddings, dim=0)
        output, _ = self.gru(x)
        h = self.norm(output[:, -1, :])
        return self.classifier(h)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_class_targets(dataset) -> np.ndarray:
    return np.asarray([label for _, label, _ in dataset.sequences], dtype=np.int64)


def batches(dataset, batch_size: int, shuffle: bool, seed: int):
    ids = list(range(len(dataset)))
    if shuffle:
        random.Random(seed).shuffle(ids)
    for start in range(0, len(ids), batch_size):
        yield ids[start:start + batch_size]


def train_three_class_epoch(model, dataset, optimizer, device, batch_size, seed, training):
    model.train(training)
    criterion = nn.CrossEntropyLoss()
    total_loss, total_n = 0.0, 0
    all_pred, all_y = [], []
    for ids in batches(dataset, batch_size, training, seed):
        graphs = [dataset[i][0] for i in ids]
        y = torch.tensor([dataset[i][1] for i in ids], dtype=torch.long, device=device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            logits = model(graphs, device)
            loss = criterion(logits, y)
            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
        total_loss += float(loss.item()) * len(ids)
        total_n += len(ids)
        all_pred.extend(logits.argmax(dim=1).detach().cpu().numpy().tolist())
        all_y.extend(y.detach().cpu().numpy().tolist())
    metrics = calculate_metrics(np.asarray(all_pred), np.asarray(all_y))
    return total_loss / max(1, total_n), metrics


def evaluate_v5(model, dataset, device, batch_size, thresholds=None):
    """Evaluate V5 without tuning on test data. Thresholds must come from validation."""
    model.eval()
    outputs = {k: [] for k in ("deadlock", "pre")}
    with torch.no_grad():
        for ids in batches(dataset, batch_size, False, 0):
            graphs = [dataset[i][0] for i in ids]
            result = model(graphs, device)
            for k in outputs:
                outputs[k].extend(torch.sigmoid(result[k]).cpu().numpy().tolist())
    y = get_class_targets(dataset)
    deadlock_p = np.asarray(outputs["deadlock"])
    pre_p = np.asarray(outputs["pre"])
    if thresholds is None:
        thresholds = tune_thresholds(deadlock_p, pre_p, y)
    pred = hierarchical_predictions(
        deadlock_p, pre_p,
        thresholds["deadlock_threshold"], thresholds["pre_threshold"]
    )
    return {
        "metrics": calculate_metrics(pred, y),
        "thresholds": {
            "deadlock_threshold": float(thresholds["deadlock_threshold"]),
            "pre_threshold": float(thresholds["pre_threshold"]),
        },
        "predictions": pred.tolist(),
        "targets": y.tolist(),
    }


def train_v5_variant(
    model,
    train_ds,
    val_ds,
    device,
    output_dir: Path,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    patience: int,
    seed: int,
    include_risk_losses: bool,
):
    """Train V5; include_risk_losses=False ablates the three risk auxiliary losses."""
    train_labels = np.asarray(train_ds.labels)
    total = len(train_labels)
    dpos = int(np.sum(train_labels == LABEL_TO_ID["deadlocked"]))
    ppos = int(np.sum(train_labels == LABEL_TO_ID["pre_deadlock"]))
    dweight = (total - dpos) / max(1, dpos)
    pweight = (total - ppos) / max(1, ppos)

    losses = {
        "deadlock": BinaryFocalLoss(pos_weight=dweight, gamma=1.0).to(device),
        "pre": BinaryFocalLoss(pos_weight=pweight, gamma=1.0).to(device),
        "risk_50": BinaryFocalLoss(pos_weight=1.0, gamma=1.0).to(device),
        "risk_100": BinaryFocalLoss(pos_weight=1.0, gamma=1.0).to(device),
        "risk_300": BinaryFocalLoss(pos_weight=1.0, gamma=1.0).to(device),
    }
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_score, best_epoch, stale = -1.0, 0, 0
    best_path = output_dir / ("v5_full.pt" if include_risk_losses else "v5_no_risk_aux.pt")
    history = []

    def run_epoch(ds, is_train, epoch):
        model.train(is_train)
        total_loss, total_n = 0.0, 0
        all_probs = {"deadlock": [], "pre": []}
        for ids in batches(ds, batch_size, is_train, seed + epoch if is_train else seed):
            graph_batch = [ds[i][0] for i in ids]
            rows = [ds[i][2] for i in ids]
            targets = {k: v.to(device) for k, v in get_targets(rows).items()}
            if is_train:
                optimizer.zero_grad(set_to_none=True)
            with torch.set_grad_enabled(is_train):
                out = model(graph_batch, device)
                ld = losses["deadlock"](out["deadlock"], targets["deadlock"])
                lp = losses["pre"](out["pre"], targets["pre"])
                loss = ld + 1.5 * lp
                if include_risk_losses:
                    loss = (
                        loss
                        + 0.2 * losses["risk_50"](out["risk_50"], targets["risk_50"])
                        + 0.3 * losses["risk_100"](out["risk_100"], targets["risk_100"])
                        + 0.7 * losses["risk_300"](out["risk_300"], targets["risk_300"])
                    )
                if is_train:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
            total_loss += float(loss.item()) * len(ids)
            total_n += len(ids)
            for k in all_probs:
                all_probs[k].extend(torch.sigmoid(out[k]).detach().cpu().numpy().tolist())

        y = get_class_targets(ds)
        dp, pp = np.asarray(all_probs["deadlock"]), np.asarray(all_probs["pre"])
        if is_train:
            thresholds = {"deadlock_threshold": 0.5, "pre_threshold": 0.5}
        else:
            thresholds = tune_thresholds(dp, pp, y)
        pred = hierarchical_predictions(
            dp, pp, thresholds["deadlock_threshold"], thresholds["pre_threshold"]
        )
        metrics = calculate_metrics(pred, y)
        safe_f1 = metrics["per_class"]["safe"]["f1"]
        pre_f1 = metrics["per_class"]["pre_deadlock"]["f1"]
        dead_f1 = metrics["per_class"]["deadlocked"]["f1"]
        score = 0.50 * pre_f1 + 0.25 * safe_f1 + 0.25 * dead_f1
        return {
            "loss": total_loss / max(1, total_n),
            "metrics": metrics,
            "thresholds": thresholds,
            "score": score,
        }

    start_time = time.time()
    for epoch in range(1, epochs + 1):
        tr = run_epoch(train_ds, True, epoch)
        va = run_epoch(val_ds, False, epoch)
        print(
            f"[{'V5-full' if include_risk_losses else 'V5-no-risk'}] "
            f"epoch {epoch:02d}/{epochs} "
            f"train_loss={tr['loss']:.4f} val_loss={va['loss']:.4f} "
            f"val_macroF1={va['metrics']['macro_f1']:.4f} "
            f"val_score={va['score']:.4f}"
        )
        history.append({
            "epoch": epoch,
            "train_loss": tr["loss"],
            "validation_loss": va["loss"],
            "validation_metrics": va["metrics"],
            "validation_thresholds": va["thresholds"],
            "selection_score": va["score"],
        })
        if va["score"] > best_score:
            best_score, best_epoch, stale = va["score"], epoch, 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "config": {
                    "input_size": 9, "hidden_size": 32,
                    "include_risk_losses": include_risk_losses,
                    **va["thresholds"],
                },
            }, best_path)
        else:
            stale += 1
        if stale >= patience:
            print(f"Early stopping at epoch {epoch}; best epoch {best_epoch}.")
            break

    history_path = output_dir / (
        "v5_full_history.json" if include_risk_losses else "v5_no_risk_aux_history.json"
    )
    history_path.write_text(json.dumps(history, indent=2))
    return {
        "checkpoint": str(best_path),
        "best_epoch": best_epoch,
        "best_validation_score": best_score,
        "training_seconds": time.time() - start_time,
        "history": history,
    }


def test_three_class(model, test_ds, device, batch_size):
    _, metrics = train_three_class_epoch(
        model, test_ds, optimizer=None, device=device,
        batch_size=batch_size, seed=0, training=False
    )
    return metrics


def load_v5_checkpoint(model, path: Path, device):
    # Allow loading checkpoints whose serialized args contain WindowsPath on macOS/Linux.
    if os.name != "nt":
        pathlib.WindowsPath = pathlib.PosixPath
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    return ckpt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path("dataset/v3"))
    parser.add_argument("--output", type=Path, default=Path("outputs/graphsagegrucomp_nofuturerisk_study"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-v5-full", action="store_true",
                        help="Use existing outputs/gnn_v5/best_model_v5.pt for full V5 in Experiment 2.")
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device={device}; torch={torch.__version__}; seed={args.seed}")
    print("Loading fixed V3 splits...")
    train_ds = TemporalSequenceDataset(args.dataset, "train")
    val_ds = TemporalSequenceDataset(args.dataset, "validation")
    test_ds = TemporalSequenceDataset(args.dataset, "test")
    print(f"Sequences: train={len(train_ds)}, validation={len(val_ds)}, test={len(test_ds)}")

    results: dict[str, Any] = {
        "metadata": {
            "dataset": str(args.dataset),
            "seed": args.seed,
            "device": str(device),
            "torch_version": torch.__version__,
            "sequence_length": 8,
            "snapshot_interval_ms": 10,
            "batch_size": args.batch_size,
            "max_epochs": args.epochs,
            "learning_rate": args.lr,
            "weight_decay": args.weight_decay,
            "patience": args.patience,
            "test_split_used_for_selection": False,
        },
        "experiment_1": {},
        "experiment_2": {},
    }

    # EXPERIMENT 1: ordinary 3-class CE/argmax classifier.
    print("\n=== EXPERIMENT 1: 3-class GraphSAGE+GRU, cross-entropy ===")
    seed_everything(args.seed)
    model1 = ThreeClassGraphSAGEGRU(9, args.hidden_size).to(device)
    opt1 = torch.optim.AdamW(model1.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_score, best_epoch, stale = -1.0, 0, 0
    best1 = args.output / "experiment1_graphsage_gru_3class.pt"
    hist1 = []
    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_metrics = train_three_class_epoch(
            model1, train_ds, opt1, device, args.batch_size, args.seed + epoch, True
        )
        va_loss, va_metrics = train_three_class_epoch(
            model1, val_ds, None, device, args.batch_size, args.seed, False
        )
        # Selection metric is Macro-F1, validation only.
        score = va_metrics["macro_f1"]
        print(f"[E1] epoch {epoch:02d}/{args.epochs} train_loss={tr_loss:.4f} "
              f"val_loss={va_loss:.4f} val_macroF1={score:.4f}")
        hist1.append({"epoch": epoch, "train_loss": tr_loss, "validation_loss": va_loss,
                      "validation_metrics": va_metrics})
        if score > best_score:
            best_score, best_epoch, stale = score, epoch, 0
            torch.save({"model_state_dict": model1.state_dict(), "epoch": epoch,
                        "config": {"input_size": 9, "hidden_size": args.hidden_size}}, best1)
        else:
            stale += 1
        if stale >= args.patience:
            break
    (args.output / "experiment1_history.json").write_text(json.dumps(hist1, indent=2))
    ckpt1 = torch.load(best1, map_location=device, weights_only=False)
    model1.load_state_dict(ckpt1["model_state_dict"])
    model1.eval()
    _, e1test = train_three_class_epoch(
        model1, test_ds, None, device, args.batch_size, args.seed, False
    )
    results["experiment_1"] = {
        "model": "GraphSAGE+GRU 3-class CE/argmax",
        "best_epoch": best_epoch,
        "best_validation_macro_f1": best_score,
        "test_metrics": e1test,
        "test_classification_report": classification_report(
            get_class_targets(test_ds),
            _predict_three_class(model1, test_ds, device, args.batch_size),
            labels=[0, 1, 2], target_names=list(LABELS), output_dict=True, zero_division=0
        ),
        "confusion_matrix": confusion_matrix(
            get_class_targets(test_ds),
            _predict_three_class(model1, test_ds, device, args.batch_size),
            labels=[0, 1, 2]
        ).tolist(),
    }

    # EXPERIMENT 2: full V5 versus V5 without auxiliary future-risk losses.
    print("\n=== EXPERIMENT 2: V5 full multitask vs no-risk-auxiliary-loss ablation ===")
    full_path = args.output / "v5_full.pt"
    if args.skip_v5_full:
        full_path = Path("outputs/gnn_v5/best_model_v5.pt")
        if not full_path.exists():
            raise FileNotFoundError(f"Requested existing V5 checkpoint not found: {full_path}")
        print(f"Using existing V5 checkpoint: {full_path}")
    else:
        seed_everything(args.seed)
        full_model = TemporalRiskAwareV5(9, args.hidden_size).to(device)
        full_train = train_v5_variant(
            full_model, train_ds, val_ds, device, args.output, args.epochs,
            args.batch_size, args.lr, args.weight_decay, args.patience,
            args.seed, include_risk_losses=True
        )
        full_path = Path(full_train["checkpoint"])
        results["experiment_2"]["full_v5_training"] = {
            k: full_train[k] for k in ("checkpoint", "best_epoch", "best_validation_score", "training_seconds")
        }

    # Train the same V5 architecture but remove risk_50/100/300 from the loss.
    seed_everything(args.seed)
    no_risk_model = TemporalRiskAwareV5(9, args.hidden_size).to(device)
    no_risk_train = train_v5_variant(
        no_risk_model, train_ds, val_ds, device, args.output, args.epochs,
        args.batch_size, args.lr, args.weight_decay, args.patience,
        args.seed, include_risk_losses=False
    )
    results["experiment_2"]["no_risk_aux_training"] = {
        k: no_risk_train[k] for k in ("checkpoint", "best_epoch", "best_validation_score", "training_seconds")
    }

    # Evaluate both variants on test with thresholds chosen on validation.
    for name, path in (("full_v5", full_path), ("v5_without_risk_auxiliary_loss", Path(no_risk_train["checkpoint"]))):
        model = TemporalRiskAwareV5(9, args.hidden_size).to(device)
        ckpt = load_v5_checkpoint(model, path, device)
        # Thresholds saved in the checkpoint came from validation.
        thresholds = {
            "deadlock_threshold": ckpt["config"]["deadlock_threshold"],
            "pre_threshold": ckpt["config"]["pre_threshold"],
        }
        ev = evaluate_v5(model, test_ds, device, args.batch_size, thresholds=thresholds)
        y, pred = np.asarray(ev["targets"]), np.asarray(ev["predictions"])
        ev["classification_report"] = classification_report(
            y, pred, labels=[0, 1, 2], target_names=list(LABELS),
            output_dict=True, zero_division=0
        )
        ev["confusion_matrix"] = confusion_matrix(y, pred, labels=[0, 1, 2]).tolist()
        results["experiment_2"][name] = ev

    out_path = args.output / "graphsagegrucomp_nofuturerisk_study_results.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nSaved all results to: {out_path}")
    print("IMPORTANT: test set was used only for final evaluation, not checkpoint/threshold selection.")


def _predict_three_class(model, dataset, device, batch_size):
    model.eval()
    pred = []
    with torch.no_grad():
        for ids in batches(dataset, batch_size, False, 0):
            graphs = [dataset[i][0] for i in ids]
            pred.extend(model(graphs, device).argmax(dim=1).cpu().numpy().tolist())
    return np.asarray(pred, dtype=np.int64)


if __name__ == "__main__":
    main()
