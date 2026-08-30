# Reproducing Dataset v3

From the repository root:

```bash
python3 scripts/generate_v3_synthetic.py \
  --output dataset/v3 \
  --runs-per-scenario 240 \
  --seed 20260830
```

Generation is deterministic for the same seed and run count.

## Scaling

`--runs-per-scenario` controls size across all 11 scenarios.

| Value | Approximate independent runs | Intended use |
|---:|---:|---|
| 12 | 132 | Fast pipeline smoke test |
| 60 | 660 | Small model experiment |
| 120 | 1,320 | Resource-constrained training |
| 240 | 2,640 | Full dataset supplied here |

Do not change the test data after observing model results. When changing the
generator or feature design, create a new dataset version and a fresh test
split. Validation and test are shifted toward larger graph sizes, but their
parameter ranges are not fully disjoint.

Wait ages obey snapshot causality: a wait first seen after snapshot zero is at
most 10 ms old, while a wait already in progress when the trace begins may have
an unknown non-zero age. Persistent waits increase by exactly 10 ms per sample.
