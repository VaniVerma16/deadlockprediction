# Detailed ablation study

**File:** `scripts/ablation_study.ipynb`

## What and why

This notebook retrains V5 after removing or shortening parts of its input. We do this to see which information helps predict deadlocks.

- **History:** compare 8, 4, and 1 snapshots. This tests whether earlier snapshots help.
- **Graph links:** remove `waits_for`, `owned_by`, or both. This tests whether link types help.
- **Scheduler features:** zero features 5–8. This tests whether runtime scheduling signals help.
- **Optional comparisons:** add results from the GraphSAGE-GRU and no-risk-loss experiments.

## How to run it

- Open the notebook in Jupyter or VS Code.
- Run cells from top to bottom.
- Set `ROOT` if the notebook cannot find this repository.
- Choose the variants in `RUN_VARIANTS`.
- Keep `EXISTING_CONTROLLED_RESULTS` as `None` unless you want to add another experiment's JSON results.

The notebook uses `dataset/v3` and `scripts/train_gnn_v5.py`. Its default variants are 4 snapshots, 1 snapshot, no `waits_for`, no `owned_by`, no graph links, and no scheduler features. Add `full_8_snapshot_reference` to include a newly trained full-input V5 model.

## How each run works

- The notebook loads the same V3 train, validation, and test splits for every variant. This keeps the data comparison consistent.
- It changes only the selected input. It keeps labels and sequence metadata unchanged.
- It resets the random seed to 123 and trains a fresh V5 model for each variant.
- Defaults are hidden size 32, batch size 32, up to 10 epochs, learning rate `0.001`, weight decay `0.0001`, and patience 3.
- It uses V5's auxiliary future-risk losses for all variants.
- It chooses the best checkpoint and prediction thresholds using validation data. This avoids using test results to tune the model.
- It evaluates the selected model once on the test split.

## Files it creates

Results go to `outputs/detailed_ablation/`.

- `detailed_ablation_results.csv` has the main test metrics for each variant.
- `detailed_ablation_results.json` also has thresholds, confusion matrices, and individual predictions.
- Each variant's subfolder has its checkpoint, training history, and evaluation file.
- Notebook cells display F1 charts and confusion matrices.
- `combined_ablation_summary.csv` is created if existing experiment results are added.

The notebook resets its output path to the shared `outputs/detailed_ablation/` folder. It does not save the seed and settings in the summary file. Record the notebook settings when saving results.

## How to read the results

- Compare Macro-F1 for the overall balance across the three classes.
- Check Pre-Deadlock precision and recall to understand false alarms versus missed warnings.
- Read confusion matrices by actual class rows and predicted class columns: Safe, Pre-Deadlock, Deadlocked.
- A single run with one seed is exploratory. Repeat with more seeds before making strong claims.
- The GraphSAGE-GRU baseline uses a different loss and prediction rule from V5. It is not an architecture-only comparison.
