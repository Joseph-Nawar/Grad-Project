# RuralStroke-Assist System Architecture

## Objective

RuralStroke-Assist is an offline-capable multimodal research prototype for stroke screening and triage support. It does not diagnose stroke.

## Phase 0 canonical architecture

Canonical modalities are face, speech, acute symptoms, and contextual metadata. The canonical fusion source is `rural_stroke_assist/modules/fusion_module.py`. The older `rural_stroke_assist/fusion/fusion_engine.py` is retained as a legacy compatibility path and uses placeholder face/speech modules.

```text
raw input and structured symptoms
  -> manifests/preprocessing
  -> branch-specific research artifacts
  -> normalized branch evidence
  -> canonical four-input fusion
  -> research screening result
```

The selected artifacts and exact hashes are recorded in `config/baseline_registry.json`.

## Current integration boundary

The trained face, speech, and metadata models are verified artifacts but are not connected to the live runtime. The acute symptom module and canonical fusion are source-level components with smoke tests; they are not yet connected to a real assessment orchestrator. Real modality adapters, end-to-end inference, and a user interface are deferred to Phase 1.

Phase 1 adapters now live under `rural_stroke_assist/inference/` and return the common immutable `ModalityEvidence` contract. `rural_stroke_assist/quality/` contains pluggable face and audio quality checks. These adapters are independently testable and artifact-backed, but are intentionally not connected to either fusion engine in this phase.

## Baseline verification

From the repository root, use the validated Python 3.11 environment and run:

```powershell
python scripts/verify_baseline.py
```

See `docs/project_audit/PHASE_0_BASELINE.md` for scope, limitations, and canonical/legacy distinctions.
