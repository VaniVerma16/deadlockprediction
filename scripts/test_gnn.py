#!/usr/bin/env python3
"""
Evaluate the trained Temporal Relational GraphSAGE + GRU model
on the held-out TEST split of dataset/v3.

IMPORTANT:
- Does NOT train the model.
- Does NOT modify best_model.pt.
- Does NOT use the training or validation split.
- Uses the exact model implementation and evaluation API
  from the current train_gnn.py.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn

from train_gnn import (
    LABELS,
    TemporalGraphClassifier,
    TemporalSequenceDataset,
    run_epoch,
)


def main() -> int:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--model",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=None,
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("=" * 70)
    print("TESTING TEMPORAL RELATIONAL GRAPHSAGE + GRU")
    print("=" * 70)

    print(f"Device: {device}")

    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    print(f"PyTorch: {torch.__version__}")

    # --------------------------------------------------------
    # Load checkpoint
    # --------------------------------------------------------

    print()
    print("Loading trained model...")
    print(f"Model: {args.model}")

    checkpoint = torch.load(
        args.model,
        map_location=device,
        weights_only=False,
    )

    # --------------------------------------------------------
    # Read configuration from checkpoint
    # --------------------------------------------------------

    config = checkpoint.get("config", {})

    hidden_size = config.get(
        "hidden_size",
        checkpoint.get("hidden_size", 32),
    )

    input_size = config.get(
        "input_size",
        9,
    )

    focal_gamma = config.get(
        "focal_gamma",
        1.5,
    )

    pre_fp_penalty = config.get(
        "pre_fp_penalty",
        0.15,
    )

    print(f"Hidden size: {hidden_size}")
    print(f"Input size: {input_size}")
    print(f"Focal gamma: {focal_gamma}")
    print(f"SAFE -> PRE penalty: {pre_fp_penalty}")

    # --------------------------------------------------------
    # Create model
    # --------------------------------------------------------

    model = TemporalGraphClassifier(
        input_size=input_size,
        hidden_size=hidden_size,
    ).to(device)

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    print("✓ Model loaded")

    # --------------------------------------------------------
    # Load TEST ONLY
    # --------------------------------------------------------

    print()
    print("Loading TEST dataset...")

    test = TemporalSequenceDataset(
        args.dataset,
        "test",
    )

    test_indices = list(range(len(test)))

    print()
    print(f"Test samples: {len(test_indices):,}")

    # --------------------------------------------------------
    # Test class distribution
    # --------------------------------------------------------

    print()
    print("Test class distribution:")

    test_counts = test.label_counts()

    for class_id, count in sorted(test_counts.items()):
        if 0 <= class_id < len(LABELS):
            label = LABELS[class_id]
        else:
            label = str(class_id)

        print(f"  {label:15s}: {count:,}")

    # --------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("TEST EVALUATION")
    print("=" * 70)

    # The test set is never used for model selection.
    # CrossEntropyLoss is sufficient here because the main purpose
    # of this script is to report test predictions and metrics.
    #
    # run_epoch() itself applies the SAFE -> PRE_DEADLOCK penalty,
    # matching the training/evaluation implementation.

    criterion = nn.CrossEntropyLoss()

    test_loss, test_metrics = run_epoch(
        model=model,
        dataset=test,
        optimizer=None,
        criterion=criterion,
        device=device,
        batch_size=args.batch_size,
        focal_gamma=focal_gamma,
        pre_fp_penalty=pre_fp_penalty,
        train=False,
        seed=42,
    )

    # --------------------------------------------------------
    # Overall results
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("FINAL TEST RESULTS")
    print("=" * 70)

    print(f"Test loss:       {test_loss:.4f}")
    print(
        f"Test accuracy:   "
        f"{test_metrics['accuracy']:.4f}"
    )
    print(
        f"Test macro-F1:   "
        f"{test_metrics['macro_f1']:.4f}"
    )
    print(
        f"Test macro-P:    "
        f"{test_metrics['macro_precision']:.4f}"
    )
    print(
        f"Test macro-R:    "
        f"{test_metrics['macro_recall']:.4f}"
    )

    # --------------------------------------------------------
    # Per-class results
    # --------------------------------------------------------

    print()
    print("Per-class results:")

    for label in LABELS:
        values = test_metrics["per_class"][label]

        print(
            f"  {label:15s} "
            f"P={values['precision']:.4f} "
            f"R={values['recall']:.4f} "
            f"F1={values['f1']:.4f} "
            f"N={values['support']}"
        )

    # --------------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------------

    print()
    print("Confusion Matrix")
    print()

    print(
        f"{'Actual / Predicted':20s}"
        f"{'safe':>10s}"
        f"{'pre_deadlock':>16s}"
        f"{'deadlocked':>14s}"
    )

    confusion = test_metrics["confusion_matrix"]

    for i, row in enumerate(confusion):
        print(
            f"{LABELS[i]:20s}"
            f"{row[0]:10d}"
            f"{row[1]:16d}"
            f"{row[2]:14d}"
        )

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    if args.output is not None:
        args.output.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        results = {
            "model": str(args.model),
            "dataset": str(args.dataset),
            "device": str(device),
            "input_size": input_size,
            "hidden_size": hidden_size,
            "focal_gamma": focal_gamma,
            "pre_fp_penalty": pre_fp_penalty,
            "test_samples": len(test_indices),
            "test_loss": test_loss,
            **test_metrics,
        }

        args.output.write_text(
            json.dumps(
                results,
                indent=2,
            ),
            encoding="utf-8",
        )

        print()
        print(f"Results saved to: {args.output}")

    print()
    print("=" * 70)
    print("TESTING COMPLETE")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
