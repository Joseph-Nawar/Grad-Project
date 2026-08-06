# Contextual metadata model card

## Intended use

Background contextual-risk ranking within the research screening prototype.

## Out of scope

Acute stroke detection, diagnosis, emergency triage by itself, or clinical probability estimation.

## Input and preprocessing

The exact ten-feature contract is age, hypertension, heart disease, average glucose, BMI, gender, ever married, work type, residence type, and smoking status. The complete saved sklearn pipeline performs imputation, scaling, and encoding.

## Score semantics

The output is `contextual_risk_evidence`. Positive class: `stroke`. It is background evidence only and excludes acute symptoms and onset.

## Data and results

The canonical manifest is [`data/processed/metadata_split_manifest.csv`](../../data/processed/metadata_split_manifest.csv), with 767 held-out rows. ROC-AUC is `0.8338`; precision is `0.1314` because the data are severely imbalanced.

## Artifact and runtime

- Artifact: [`mvp_metadata_risk_model.pkl`](../../models/experiments/metadata/mvp_metadata_risk_model.pkl)
- SHA256: `350545A6AB71A58373FD11BC2EF9F37BAADCC8420D66186EB6F4BD2B8AA1E8F5`
- Size: 6,407 bytes
- Evidence: [`metadata_context.json`](../../reports/evaluation/phase4/final_complete/metadata_context.json)

## Limitations

The manifest is a general-risk dataset rather than paired acute-triage data. Calibration and positive precision are limited. Stored artifact/runtime version comparison remains pending unless an external scikit-learn 1.4.2 interpreter is supplied. Results are contextual evidence, not clinical prediction.
