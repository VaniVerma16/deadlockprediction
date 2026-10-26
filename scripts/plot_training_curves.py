#!/usr/bin/env python3
"""Plot V5 training/validation curves from training_history_v5.json.

Usage (from the repository root):
    python kernel_deadlock_prototype/experiments/plot_training_curves.py
    python kernel_deadlock_prototype/experiments/plot_training_curves.py \
        --history outputs/gnn_v5/training_history_v5.json --out figures

Writes (PNG at 300 dpi and vector PDF for each):
    fig_loss          train vs validation loss
    fig_accuracy      validation accuracy
    fig_macro_f1      validation macro-F1
    fig_class_f1      validation F1 per class
    fig_risk_f1       validation F1 of the 50/100/300 ms risk heads
    fig_training_overview   all of the above in one 2x3 panel

Requires: matplotlib (pip install matplotlib)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Paper-friendly defaults (serif, small text, no top/right spines).
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "legend.fontsize": 7,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linewidth": 0.5,
    "figure.dpi": 150,
})

TRAIN_COLOR = "#1f77b4"
VAL_COLOR = "#d62728"
CLASS_COLORS = {"safe": "#2ca02c", "pre_deadlock": "#ff7f0e", "deadlocked": "#d62728"}
CLASS_LABELS = {"safe": "Safe", "pre_deadlock": "Pre-Deadlock", "deadlocked": "Deadlocked"}
RISK_COLORS = {"risk_50": "#1f77b4", "risk_100": "#9467bd", "risk_300": "#8c564b"}
RISK_LABELS = {"risk_50": "50 ms", "risk_100": "100 ms", "risk_300": "300 ms"}

FIG_SIZE = (3.4, 2.4)  # inches: one column of a two-column paper


def load(path: Path) -> list[dict]:
    history = json.loads(path.read_text())
    if not isinstance(history, list) or not history:
        raise SystemExit(f"{path}: expected a non-empty list of per-epoch records")
    return history


def best_epoch(history: list[dict]) -> int:
    """Epoch chosen by the validation selection score (the saved checkpoint)."""
    return max(history, key=lambda e: e["selection_score"])["epoch"]


def mark_best(ax, epoch: int) -> None:
    ax.axvline(epoch, color="gray", linestyle="--", linewidth=0.8, label=f"Best epoch ({epoch})")


def style_epoch_axis(ax, epochs: list[int]) -> None:
    ax.set_xlabel("Epoch")
    ax.set_xlim(min(epochs), max(epochs))
    ax.set_xticks([e for e in epochs if e == 1 or e % 5 == 0])


def panel_train_val(ax, history, epochs, getter, ylabel, best):
    ax.plot(epochs, [getter(e["train"]) for e in history], color=TRAIN_COLOR,
            marker="o", markersize=2.5, linewidth=1.2, label="Train")
    ax.plot(epochs, [getter(e["validation"]) for e in history], color=VAL_COLOR,
            marker="s", markersize=2.5, linewidth=1.2, label="Validation")
    ax.set_ylabel(ylabel)
    mark_best(ax, best)
    style_epoch_axis(ax, epochs)
    ax.legend(loc="best", frameon=False)


def panel_val_only(ax, history, epochs, getter, ylabel, best):
    ax.plot(epochs, [getter(e["validation"]) for e in history], color=VAL_COLOR,
            marker="s", markersize=2.5, linewidth=1.2, label="Validation")
    ax.set_ylabel(ylabel)
    mark_best(ax, best)
    style_epoch_axis(ax, epochs)
    ax.legend(loc="lower right", frameon=False)


def panel_class_f1(ax, history, epochs, best):
    for key in ("safe", "pre_deadlock", "deadlocked"):
        ax.plot(epochs, [e["validation"]["metrics"]["per_class"][key]["f1"] for e in history],
                color=CLASS_COLORS[key], marker="o", markersize=2.5, linewidth=1.2,
                label=CLASS_LABELS[key])
    ax.set_ylabel("Validation F1")
    ax.set_ylim(0, 1.0)
    mark_best(ax, best)
    style_epoch_axis(ax, epochs)
    ax.legend(loc="lower right", frameon=False)


def panel_risk_f1(ax, history, epochs, best):
    for key in ("risk_50", "risk_100", "risk_300"):
        ax.plot(epochs, [e["validation"]["risk_metrics"][key]["f1"] for e in history],
                color=RISK_COLORS[key], marker="o", markersize=2.5, linewidth=1.2,
                label=RISK_LABELS[key])
    ax.set_ylabel("Validation risk F1")
    ax.set_ylim(0, 1.0)
    mark_best(ax, best)
    style_epoch_axis(ax, epochs)
    ax.legend(loc="lower right", frameon=False, title="Horizon", title_fontsize=7)


PANELS = {
    "fig_loss": lambda ax, h, ep, b: panel_train_val(ax, h, ep, lambda s: s["loss"], "Loss", b),
    "fig_accuracy": lambda ax, h, ep, b: panel_val_only(
        ax, h, ep, lambda s: s["metrics"]["accuracy"], "Validation accuracy", b),
    "fig_macro_f1": lambda ax, h, ep, b: panel_val_only(
        ax, h, ep, lambda s: s["metrics"]["macro_f1"], "Validation macro-F1", b),
    "fig_class_f1": panel_class_f1,
    "fig_risk_f1": panel_risk_f1,
}
TITLES = {
    "fig_loss": "(a) Loss",
    "fig_accuracy": "(b) Validation accuracy",
    "fig_macro_f1": "(c) Validation macro-F1",
    "fig_class_f1": "(d) Per-class validation F1",
    "fig_risk_f1": "(e) Future-risk heads",
}


def save(fig, out_dir: Path, name: str) -> None:
    for ext in ("png", "pdf"):
        path = out_dir / f"{name}.{ext}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
    print(f"wrote {out_dir / name}.png / .pdf")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--history", type=Path, default=Path("../outputs/gnn_v5/training_history_v5.json"))
    parser.add_argument("--out", type=Path, default=Path("figures"))
    args = parser.parse_args()

    history = load(args.history)
    epochs = [e["epoch"] for e in history]
    best = best_epoch(history)
    args.out.mkdir(parents=True, exist_ok=True)

    # Individual single-column figures.
    for name, draw in PANELS.items():
        fig, ax = plt.subplots(figsize=FIG_SIZE)
        draw(ax, history, epochs, best)
        ax.set_title(TITLES[name])
        fig.tight_layout()
        save(fig, args.out, name)
        plt.close(fig)

    # Combined 2x3 overview (last cell: selection score and time per epoch).
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.4))
    for ax, name in zip(axes.flat, PANELS):
        PANELS[name](ax, history, epochs, best)
        ax.set_title(TITLES[name])
    ax = axes.flat[5]
    ax.plot(epochs, [e["selection_score"] for e in history], color="#333333",
            marker="o", markersize=2.5, linewidth=1.2)
    ax.set_ylabel("Selection score")
    ax.set_title("(f) Checkpoint selection score")
    mark_best(ax, best)
    style_epoch_axis(ax, epochs)
    ax.legend(loc="lower right", frameon=False)
    fig.tight_layout()
    save(fig, args.out, "fig_training_overview")
    plt.close(fig)

    total = sum(e["epoch_seconds"] for e in history)
    print(f"\nepochs: {len(history)}   best epoch: {best}   "
          f"total training time: {total:,.0f} s ({total / 3600:.2f} h)")
    print("Note: only loss is plotted for train vs validation. Training accuracy/F1 in the history "
          "use fixed 0.5 thresholds with class-weighted loss, so they are not comparable with the "
          "threshold-tuned validation metrics and are deliberately not plotted.")


if __name__ == "__main__":
    main()