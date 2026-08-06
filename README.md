# RuralStroke-Assist

## 1. Problem

RuralStroke-Assist is a research prototype for multimodal stroke screening and triage support in settings where a specialist may not be immediately available. It combines facial-image evidence, speech evidence, acute symptoms, and contextual metadata into a structured evidence result. It is not a diagnostic system and its scores are not clinical probabilities.

## 2. Product demonstration

The project demonstrates a local two-role workflow:

1. A rural collector starts a draft, records contextual metadata, FAST/BE-FAST symptoms, a facial image, and an English speech sample.
2. The collector runs the existing `AssessmentService`, reviews evidence, limitations, and warnings, then submits the immutable assessment snapshot.
3. A clinician opens the submitted-case queue, reviews the exact snapshot and submitted media, and records whether they agree with or override the proposed triage urgency.

The applications are local Streamlit demonstrations. They do not diagnose patients, synchronize to a cloud service, or provide production authentication.

## 3. Architecture

The active path is:

```text
Collector app -> case workflow -> AssessmentService
  -> face / speech / metadata / acute-symptom adapters
  -> canonical late fusion -> immutable assessment snapshot
  -> SQLite case repository and managed attachments
  -> Clinician app -> separate agreement or override review
```

The canonical fusion implementation is [`rural_stroke_assist/modules/fusion_module.py`](rural_stroke_assist/modules/fusion_module.py), wrapped by [`rural_stroke_assist/assessment/fusion_strategy.py`](rural_stroke_assist/assessment/fusion_strategy.py). The older [`rural_stroke_assist/fusion/fusion_engine.py`](rural_stroke_assist/fusion/fusion_engine.py) is legacy and is outside the active path.

See the [system architecture diagram](docs/diagrams/system_architecture.md) and [case data flow diagram](docs/diagrams/case_data_flow.md).

## 4. Technical contribution

The implementation contribution is an offline-capable, typed inference and workflow boundary around heterogeneous research artifacts. It provides:

- immutable modality evidence and assessment contracts;
- lazy registry-backed model loading;
- explicit quality rejection and missing-modality semantics;
- deterministic four-input fusion with per-modality contributions;
- immutable submitted assessment snapshots;
- replaceable case persistence and managed attachments;
- separate clinician review decisions;
- reproducible held-out proxy evaluation and robustness evidence.

## 5. Results

<!-- PHASE4_RESULTS_START -->
These values are sourced from [`reports/evaluation/phase4/final_complete`](reports/evaluation/phase4/final_complete), not from notebooks or the superseded Phase 4 run.

### Direct canonical held-out evaluation

- Face: 318 test images, ROC-AUC `0.9818`, sensitivity `0.9000`, specificity `0.9471`.
- Speech: 2,849 speaker-held-out recordings, ROC-AUC `0.9426279894`.
- Metadata: 767 test rows, ROC-AUC `0.8338`; precision is `0.1314`, so this branch remains contextual-risk ranking evidence only.

### Runtime-adapter evaluation

- Face adapter: 107 accepted of 318 images, coverage `33.65%`; accepted-subset ROC-AUC `0.9607`.
- Speech adapter: 2,744 accepted of 2,849 recordings, coverage `96.31%`; accepted-subset ROC-AUC `0.9512`.
- The historical speech ROC-AUC `0.9426` is the direct full-test result. The `0.9512` result is the quality-filtered accepted subset.

The face adapter coverage limitation is prominent: direct model performance must not be interpreted as performance over the runtime population because most held-out face inputs are rejected by quality checks.

Acute symptoms are evaluated through deterministic rule verification, not classifier accuracy. Fusion and end-to-end outputs are engineering robustness and sensitivity results only. No paired multimodal clinical dataset exists, so end-to-end diagnostic accuracy and fusion clinical benefit are unsupported.
<!-- PHASE4_RESULTS_END -->

## 6. Limitations

- Face, speech, and metadata branches use public proxy data rather than paired clinical stroke data.
- TORGO dysarthria is not stroke-specific.
- Metadata represents background risk factors, not acute stroke presentation.
- Face runtime coverage is only `33.65%` on the held-out split and is class-conditionally uneven.
- Scores are branch-specific evidence or contextual-risk scores, not calibrated clinical stroke probabilities.
- Clinical wording approval remains pending human review.
- Local case storage has no production authentication, encryption, remote synchronization, or compliance controls.

## 7. Setup

The validated baseline uses Python 3.11 and the repository-local bootstrap route:

```powershell
py -3.11 scripts/bootstrap_local.py
```

For an existing validated environment, run:

```powershell
python scripts/verify_baseline.py
python -m pytest -q
```

Launch the two applications with the canonical launcher:

```powershell
python scripts/run_phase3_apps.py
```

Run the authoritative evaluation with:

```powershell
python scripts/run_phase4_evaluation.py --suite full --output-dir reports/evaluation/phase4/final_complete
```

The evaluation output directory refuses overwrite unless `--overwrite` is explicitly supplied.

## 8. Repository structure

- [`rural_stroke_assist/inference`](rural_stroke_assist/inference) — modality adapters and quality checks.
- [`rural_stroke_assist/assessment`](rural_stroke_assist/assessment) — assessment orchestration and canonical fusion wrapper.
- [`rural_stroke_assist/cases`](rural_stroke_assist/cases) — case lifecycle, persistence, attachments, and reports.
- [`apps`](apps) — collector and clinician Streamlit entrypoints.
- [`rural_stroke_assist/evaluation`](rural_stroke_assist/evaluation) — offline evaluation framework.
- [`models/experiments`](models/experiments) — saved research artifacts; see [`config/baseline_registry.json`](config/baseline_registry.json).
- [`data/processed`](data/processed) — canonical manifests and processed experiment data.
- [`reports/evaluation/phase4/final_complete`](reports/evaluation/phase4/final_complete) — authoritative Phase 4 evidence.
- [`docs`](docs) — architecture, cards, reproducibility, testing, privacy, demo, and release documentation.
- [`config/repository_catalog.yaml`](config/repository_catalog.yaml) — active, auxiliary, legacy, and historical classification.

## Testing

See [`docs/testing/TESTING.md`](docs/testing/TESTING.md) for fast, integration, evaluation, and release-validation tiers.

## Evaluation

See [`docs/reproducibility/REPRODUCIBILITY.md`](docs/reproducibility/REPRODUCIBILITY.md) and the [Phase 4 final report](reports/evaluation/phase4/final_complete/PHASE_4_EVALUATION_REPORT.md). Evaluation results are non-clinical proxy and engineering evidence.

## Privacy

See [`docs/privacy/PRIVACY_AND_DATA_RETENTION.md`](docs/privacy/PRIVACY_AND_DATA_RETENTION.md) before using real media or identifiers. Demonstrations should use pseudonymous public-data cases only.

## Documentation and release

The release is `0.0.1`, labelled **Research MVP — not clinically validated**. See [`docs/release/RELEASE_CHECKLIST.md`](docs/release/RELEASE_CHECKLIST.md) and [`reports/release/release_manifest.json`](reports/release/release_manifest.json).
