# Metadata pretrained foundation trial 001

This directory contains the standalone research-only `validate -> select -> test`
experiment for the ten-field metadata/context branch. The branch remains
contextual/background-risk evidence; this trial does not diagnose acute stroke
or modify canonical inference, fusion, APIs, deployment profiles, or the
canonical Logistic Regression artifact.

## Reproduce

Use the dedicated `.metadata-pretrained-experiment.venv` environment and the
pinned requirements file at the repository root:

```powershell
.metadata-pretrained-experiment.venv\Scripts\python.exe -m pip check
.metadata-pretrained-experiment.venv\Scripts\python.exe scripts\metadata_pretrained_foundation_experiment.py validate
.metadata-pretrained-experiment.venv\Scripts\python.exe scripts\metadata_pretrained_foundation_experiment.py select
.metadata-pretrained-experiment.venv\Scripts\python.exe scripts\metadata_pretrained_foundation_experiment.py test
```

`validate` loads only the train and validation rows. It writes the frozen
protocol before loading either partition, replays the canonical Logistic
Regression only on validation, and records terminal `completed` or `failed`
evidence for both pretrained candidates. `select` verifies those hashes and
freezes the validation-PR-AUC decision. The inclusive 0.01 absolute PR-AUC
near-tie rule invokes the predeclared checkpoint-size, p50-latency, peak-RSS,
then candidate-ID tie-break. `test` is one-time, verifies the immutable
selection evidence, and evaluates only the selected pretrained candidate.

The runner intentionally does not freshly replay the Logistic Regression on
test. Phase 4 baseline test metrics are historical/previously exposed context
only.

## Outputs

`REPORT.md` is the human-readable closeout. JSON/CSV/PNG files contain the
validation comparison, selection snapshot, selected-candidate test metrics,
bootstrap intervals, calibration/PR/ROC outputs, runtime/resource measurements,
provenance, and non-recursive SHA-256 artifact manifest. Upstream model
checkpoints remain in the library cache and are not committed here.
