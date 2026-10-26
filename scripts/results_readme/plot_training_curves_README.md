# Plot V5 training curves

**File:** `scripts/plot_training_curves.py`

## What and why

This script turns V5's saved training history into figures. The curves help show whether training improves, whether validation scores stall, and which classes or risk horizons are hardest to predict.

## How to run it

From the repository root:

```bash
python3 scripts/plot_training_curves.py \
  --history outputs/gnn_v5/training_history_v5.json \
  --out outputs/v5_analysis/training_curves
```

The history file must come from V5 training and contain one record per epoch. Install Matplotlib if it is missing. Use the explicit `--history` path so the script reads the intended run.

## Figures it creates

Each figure is saved as a 300-DPI PNG and a vector PDF in the output folder.

- `fig_loss`: train and validation loss. This helps spot overfitting.
- `fig_accuracy`: validation accuracy.
- `fig_macro_f1`: validation Macro-F1 across all classes.
- `fig_class_f1`: validation F1 for Safe, Pre-Deadlock, and Deadlocked.
- `fig_risk_f1`: validation F1 for 50, 100, and 300 ms risk predictions.
- `fig_training_overview`: all five plots plus the checkpoint selection score.

The dashed line marks the epoch with the highest validation selection score. This is the epoch used for checkpoint selection.

## How to read the plots

- Loss is shown for both training and validation. A growing gap can indicate overfitting.
- Accuracy and F1 plots show validation results only.
- Training classification scores use fixed thresholds. Validation scores use tuned thresholds. The script omits training accuracy and F1 because those scores are not directly comparable.
- Risk F1 shows how well each auxiliary head predicts deadlock within its time horizon.
- The console summary reports the best epoch and total training time.

The plotter expects V5's history fields, including `train.loss`, `validation.metrics`, `validation.risk_metrics`, `selection_score`, and `epoch_seconds`. Histories from the other comparison script use a different format.
