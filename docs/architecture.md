# RuralStroke-Assist System Architecture

## Objective

RuralStroke-Assist is an offline-capable multimodal research prototype for stroke screening and triage support. It does not diagnose stroke.

## Phase 0 canonical architecture

Canonical modalities are face, speech, acute symptoms, and contextual metadata. The canonical fusion source is `rural_stroke_assist/modules/fusion_module.py`. The older `rural_stroke_assist/fusion/fusion_engine.py` is retained as a legacy compatibility path with its older result contract and weights.

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

The Phase 1 adapters are artifact-backed and independently testable. Phase 2 adds the sequential `AssessmentService`, but no UI, API, deployment layer, or new model.

Phase 1 adapters live under `rural_stroke_assist/inference/` and return the common immutable `ModalityEvidence` contract. `rural_stroke_assist/quality/` contains pluggable face and audio quality checks. Phase 2 `rural_stroke_assist/assessment/` composes those adapters with `CanonicalLateFusionStrategy`, which wraps `rural_stroke_assist/modules/fusion_module.py`. The legacy `rural_stroke_assist/fusion/fusion_engine.py` is not imported by the new service and emits a deprecation warning when called.

The executable developer smoke path is `python scripts/run_assessment_smoke.py`. It runs all four modalities against canonical manifest examples and prints only structured scores, bands, quality states, explanations, and timings; it does not expose input paths.

## Phase 3 case workflow

`rural_stroke_assist/cases/` is the replaceable case/workflow layer. `CaseWorkflowService` is the only component that calls `AssessmentService`; the collector creates and assesses drafts, while the clinician application reads the immutable submitted assessment snapshot and records a separate review decision. `SQLiteCaseRepository` stores JSON snapshots and audit events with WAL mode and parameterized transactions. `AttachmentStore` keeps generated relative references under ignored `runtime_data/cases/<case-id>/`.

The role-specific entrypoints are `apps/collector_app.py` and `apps/clinician_app.py`. Launch both with `python scripts/run_phase3_apps.py`, or run each Streamlit app individually on ports 8501 and 8502. This is a local workflow demonstration only; it has no API, remote synchronization, production authentication, or deployment infrastructure.

The Streamlit presentation layer is centralized in `rural_stroke_assist/ui/presentation.py` and `rural_stroke_assist/ui/common.py`. It humanizes stored fields, groups warnings, displays evidence-band guidance, and resolves managed media without changing assessment snapshots.

## Baseline verification

From the repository root, use the validated Python 3.11 environment and run:

```powershell
python scripts/verify_baseline.py
```

See `docs/project_audit/PHASE_0_BASELINE.md` for scope, limitations, and canonical/legacy distinctions.
## Phase 4 evaluation

The isolated `rural_stroke_assist.evaluation` package evaluates the canonical
held-out proxy partitions and deterministic scenario behavior. Run
`python scripts/run_phase4_evaluation.py --suite smoke` for a fast check, or
`--suite modality`, `--suite system`, and `--suite full` for progressively
broader runs. Outputs are immutable run directories under
`reports/evaluation/phase4/`. These results are engineering evidence only;
the repository has no paired multimodal clinical dataset and therefore does
not report end-to-end diagnostic accuracy or fusion clinical improvement.
