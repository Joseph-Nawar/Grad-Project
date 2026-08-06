# Face model card

## Intended use

Research-only visual proxy evidence for the RuralStroke-Assist face branch.

## Out of scope

Diagnosis, clinical probability, facial-palsy diagnosis, emergency disposition, or use outside the demonstrated research workflow.

## Input and preprocessing

RGB image converted, resized to 160×160, and scaled according to the canonical model contract. The runtime adapter additionally performs face presence, count, size, dimension, blur, and lighting checks.

## Score semantics

Sigmoid Stroke-class score interpreted as `visual_proxy_evidence`, not a calibrated clinical probability. Positive class: `Stroke`.

## Data and direct results

The canonical manifest is [`data/processed/face_split_manifest.csv`](../../data/processed/face_split_manifest.csv). The held-out test partition contains 318 images. Direct ROC-AUC is `0.9818`, sensitivity `0.9000`, and specificity `0.9471`.

## Runtime adapter coverage

The adapter accepted 107/318 images (`33.65%`) and rejected 211. Accepted-subset ROC-AUC is `0.9607`. Rejections are documented in [`face_adapter_coverage.json`](../../reports/evaluation/phase4/final_complete/face_adapter_coverage.json) and [`face_rejection_by_reason.csv`](../../reports/evaluation/phase4/final_complete/face_rejection_by_reason.csv).

## Artifact and runtime

- Artifact: [`model.keras`](../../models/experiments/face/trial_003_mobilenetv2_balanced_160/model.keras)
- SHA256: `C969C473CD369AEFE11815208407320D559AA7683998CA0EE28C53280C7011AC`
- Size: 9,641,144 bytes
- Phase 4 evidence: [`face.json`](../../reports/evaluation/phase4/final_complete/face.json)

## Limitations

This is public proxy image data with no clinical validation and no guarantee that the image label represents acute facial weakness. Runtime face coverage is low and class-conditionally uneven. All results are non-diagnostic.
