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

The saved face, speech, and metadata models are verified research artifacts but are not connected to the live runtime yet. The canonical post-MVP fusion source is `rural_stroke_assist/modules/fusion_module.py`; `rural_stroke_assist/fusion/fusion_engine.py` is retained as a legacy placeholder path. Real adapters, end-to-end inference, and a user interface are Phase 1 work.

Phase 1 provides independently testable artifact-backed adapters in `rural_stroke_assist/inference/` for face, speech, contextual metadata, and acute symptoms. Run their tests with `python -m pytest tests/test_inference_adapters.py tests/test_inference_integration.py tests/test_quality_assessment.py -q`. The adapters do not perform fusion or orchestration.
