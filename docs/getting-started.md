# Getting started

## Phase 0 baseline environment

Use Python `3.11.9`. The validated local environment is `../autokeras_env` relative to the repository directory. A clean environment can be provisioned with:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-baseline.txt
```

No dataset download is required for baseline verification, but the canonical model and processed manifest artifacts must be present locally.

## Verify the baseline

From the repository root:

```powershell
python scripts/verify_baseline.py
python -m pytest -q
python -m compileall -q rural_stroke_assist scripts tests
```

The verifier checks `config/baseline_registry.json`, artifact hashes, model loading and structure, feature contracts, acute symptom behavior, canonical fusion defaults, and manifest hashes. It does not train, regenerate, or overwrite artifacts.

## Current runtime boundary

The saved face, speech, and metadata models are verified research artifacts used by the Phase 1 adapters and Phase 2 assessment service. The canonical fusion source is `rural_stroke_assist/modules/fusion_module.py`; `rural_stroke_assist/fusion/fusion_engine.py` is retained as a legacy path and warns when called. No UI or API is included in Phase 2.

Phase 1 provides independently testable artifact-backed adapters in `rural_stroke_assist/inference/` for face, speech, contextual metadata, and acute symptoms. Phase 2 provides `create_default_assessment_service()` and `AssessmentService.assess(...)` in `rural_stroke_assist/assessment/`. Verify the real path with `python scripts/run_assessment_smoke.py`; it returns research `evidence_score` and risk-band output, never a clinical probability.

## Phase 3 local workflow

Install the pinned Streamlit dependency and launch both role-specific applications:

```powershell
python -m pip install -r requirements-baseline.txt
python scripts/run_phase3_apps.py
```

The collector is available at `http://localhost:8501` and the clinician review app at `http://localhost:8502`. Individual commands are:

```powershell
python -m streamlit run apps/collector_app.py --server.port 8501
python -m streamlit run apps/clinician_app.py --server.port 8502
```

Cases and managed attachments are stored locally under `runtime_data/`, which is ignored by the repository. The clinician app never loads models or reruns an assessment; it reviews the submitted immutable result and records agreement or override separately.

The collector displays a language limitation notice for the English TORGO speech branch. The clinician page presents human-readable context, symptoms, media, modality summaries, grouped limitations, and collapsed developer/technical details.
## Reproducible evaluation

Using the validated `autokeras_env`, run:

```powershell
python scripts/run_phase4_evaluation.py --suite smoke
```

Use `--suite modality`, `--suite system`, or `--suite full` for broader
evaluation. A non-empty output directory is never overwritten unless
`--overwrite` is explicitly supplied. Phase 4 evaluates held-out proxy data,
deterministic robustness, and engineering sensitivity; it is not clinical
validation and does not calculate paired end-to-end stroke accuracy.
