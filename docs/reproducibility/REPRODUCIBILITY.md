# Reproducibility

## Baseline

Use Python 3.11 in the validated `autokeras_env` or a clean repository-local `.venv`. Run:

```powershell
python scripts/verify_baseline.py
python -m pytest -q
python -m compileall -q rural_stroke_assist apps scripts tests
```

The baseline verifier checks registry structure, artifact and manifest SHA256 values, model loading, feature contracts, symptoms, and canonical fusion defaults.

## Evaluation

The authoritative evaluation is:

```powershell
python scripts/run_phase4_evaluation.py --suite full --output-dir reports/evaluation/phase4/final_complete
```

Phase 4 uses seed 42 and 1,000 bootstrap iterations for full runs. Smoke runs use five samples and 25 iterations. Typical full evaluation time is several minutes because the speech split and adapter audits are executed. Outputs include the run manifest, metrics, plots, rejection audits, robustness matrices, profiling, claims, and report.

The metadata artifact was serialized with scikit-learn 1.4.2 and is loaded in the validated 1.5.2 runtime, producing an expected warning. Cross-version comparison is optional and is skipped unless a separately provisioned external interpreter is supplied.
