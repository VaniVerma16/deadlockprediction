# GraphSAGE-GRU and future-risk comparison

**File:** `scripts/graphsagegrucomp_nofuturerisk_study.py`

## What and why

This script runs two experiments on the V3 dataset.

- **Three-class GraphSAGE-GRU:** predicts Safe, Pre-Deadlock, or Deadlocked with cross-entropy and argmax. This gives a standard model baseline.
- **Full V5 vs. V5 without risk losses:** trains the same V5 model with and without the 50, 100, and 300 ms future-risk losses. This tests whether those training signals help state prediction.

Both models use 8 snapshots, 10 ms apart, and 9 node features. The standard model encodes each graph with GraphSAGE, passes the sequence through a GRU, then predicts a class. V5 uses separate deadlock and pre-deadlock heads. It also has future-risk heads.

## How to run it

From the repository root:

```bash
python3 scripts/graphsagegrucomp_nofuturerisk_study.py --dataset dataset/v3
```

Default settings are seed 42, hidden size 32, batch size 32, up to 20 epochs, learning rate `0.001`, weight decay `0.0001`, and patience 5. Use `--skip-v5-full` to load `outputs/gnn_v5/best_model_v5.pt` instead of retraining full V5.

## How each run works

- The script loads the fixed V3 train, validation, and test splits.
- It trains the three-class baseline. Validation Macro-F1 selects its checkpoint.
- It trains full V5, unless `--skip-v5-full` is set.
- It trains V5 again without the auxiliary risk losses. Other V5 training losses remain enabled.
- Validation selects each V5 checkpoint and chooses its state prediction thresholds.
- It evaluates the selected models on the test split. Test results do not affect model or threshold selection.

The checked-in results record seed 42, CPU, PyTorch 2.2.2, and the default training settings. Different hardware or software may give slightly different results.

## Files it creates

The default output folder is `outputs/graphsagegrucomp_nofuturerisk_study/`.

- `experiment1_graphsage_gru_3class.pt` and `experiment1_history.json` are the baseline checkpoint and training history.
- `v5_full.pt` and `v5_full_history.json` are created when full V5 is trained.
- `v5_no_risk_aux.pt` and `v5_no_risk_aux_history.json` are the no-risk-loss checkpoint and history.
- `graphsagegrucomp_nofuturerisk_study_results.json` contains run settings and final test metrics.

The results JSON also includes classification reports, confusion matrices, and the validation thresholds used for V5.

## How to read the results

- Compare full V5 with V5 without risk losses to measure the effect of auxiliary risk training.
- Compare class precision, recall, and F1. Pre-Deadlock recall shows how many warning cases the model finds.
- Treat the three-class model as a broad baseline. Its loss and prediction rule differ from V5.
- One seed is not enough to show that a small score difference is reliable.
