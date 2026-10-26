# Model comparison notebook

**File:** `scripts/model_comparison.ipynb`

## What and why

This notebook compares several models on the synthetic V3 deadlock dataset. We use it to see how simple baselines and neural models perform beside the saved V5 model.

- **Persistence:** predicts the state seen in snapshot S6. This is a simple reference point.
- **Logistic Regression, Random Forest, and optional XGBoost:** use compact features from the latest snapshot.
- **GRU and GraphSAGE-GRU:** use the sequence of snapshots S0–S7. These test the value of time history.
- **GraphSAGE:** uses snapshot S7 only. This tests graph information without a sequence model.
- **Saved V5:** uses the trained risk-aware GraphSAGE-GRU checkpoint.

All models predict the sequence label. The notebook also checks which models detect changes from S6 to S7 and whether they warn on worsening states.

## How to run it

- Open the notebook in Jupyter or VS Code.
- Run every cell from top to bottom.
- Make sure `dataset/v3` and `outputs/gnn_v5/best_model_v5.pt` exist.
- If paths are wrong, update `REPO_ROOT`, `DATASET_DIR`, and `CHECKPOINT_PATH` in the setup cell.
- XGBoost is optional. Other models can run without it.

The notebook uses seed 42. It trains the baseline models on the train split and uses validation Macro-F1 to keep the best neural model weights. The GRU runs up to 10 epochs with batch size 256. Graph models run up to 5 epochs with batch size 32. They use AdamW, cross-entropy, and gradient clipping. V5 is loaded from disk; this notebook does not retrain it. V5 uses the thresholds saved in its checkpoint.

## What it measures

- It reports accuracy, Macro-F1, and per-class precision, recall, and F1 on test.
- It checks V5 risk predictions for deadlock within 50, 100, and 300 ms.
- It compares predictions with the previous state at S6 to measure state-change detection and dangerous transitions.
- The risk-head metrics use a probability threshold of 0.5. The V5 state thresholds come from the saved checkpoint.

## Files it creates

Results go to `outputs/v5_analysis/`.

- `model_comparison.csv` contains test scores for each model.
- `transition_metrics.csv` contains state-change and dangerous-transition scores.
- `v5_analysis_report.json` records V5 thresholds and test metrics.
- `v5_summary.png` shows the V5 confusion matrix and risk-head F1.
- `observed_transitions.png` shows actual S6-to-S7 state changes.
- `ablation_macro_f1.png` compares model Macro-F1.
- `state_change_confusion_matrices_only.png` focuses on examples where the state actually changed.

The notebook also displays tables and figures as it runs. The state-change-only figure uses fewer examples than the full test results, so its counts cannot be compared directly with the full confusion matrix.

## How to read the results

- Macro-F1 balances scores across Safe, Pre-Deadlock, and Deadlocked.
- Check Pre-Deadlock recall to see how many early warnings are found.
- Check transition metrics to see whether models notice worsening states.
- V3 is synthetic data. These results do not show how the models perform on production telemetry.
- Models use different inputs and training methods. Treat the comparison as a broad comparison, not a perfectly controlled architecture test.
