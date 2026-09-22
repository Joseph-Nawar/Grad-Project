# Metadata Pretrained Foundation Trial 001

This is a research-only evaluation of contextual/background-risk evidence from structured metadata. It does not diagnose acute stroke, change fusion, or modify production inference.

## Baseline

The frozen Logistic Regression artifact was replayed on validation only. Its Phase 4 test metrics are shown only as historical/previously exposed context; this trial did not freshly load or evaluate Logistic Regression on test.

Historical Phase 4 baseline test ROC-AUC: 0.833767; PR-AUC: 0.185651; precision: 0.131356.
Current-trial validation replay at the fixed 0.5 threshold: accuracy=0.737598; balanced_accuracy=0.746691; brier=0.170732; calibration_error=0.272831; f1=0.217899; n=766; negative=729; positive=37; positive_prevalence=0.048303; pr_auc=0.296577; precision=0.127273; roc_auc=0.845179; sensitivity=0.756757; specificity=0.736626; threshold=0.500000.

## Pretrained candidates

TabPFN v2 was explicitly selected through ModelVersion.V2 with canonical categorical positions and the CPU pretraining-limit override recorded when applicable. TabICLv2 used the pinned classification checkpoint `tabicl-classifier-v2-20260212.ckpt`. Neither candidate was fine-tuned, oversampled, or broadly searched.

## Validation comparison

Validation PR-AUC/average precision is primary because the positive class is rare. ROC-AUC can remain high while precision and PR-AUC remain modest because most thresholded positives can be false positives under severe imbalance.

| Candidate | Status | Validation PR-AUC | Validation ROC-AUC | N | Positive prevalence |
|---|---|---:|---:|---:|---:|
| logistic_regression | completed | 0.29657693527785456 | 0.8451785118451786 | 766 | 0.048302872062663184 |
| tabpfn_v2 | completed | 0.3150199238865652 | 0.8434360286212138 | 766 | 0.048302872062663184 |
| tabicl_v2 | completed | 0.2725167637371232 | 0.840470099729359 | 766 | 0.048302872062663184 |

## Validation decision

Selected pretrained candidate: `tabpfn_v2`. The decision used validation PR-AUC first, the inclusive 0.01 absolute near-tie margin, and the predeclared resource tie-break. Selection was frozen before test access.

Terminal evidence for both pretrained candidates is hash-bound in `selection_frozen.json`; failed candidates remain explicit evidence rather than being silently omitted.

Rejected pretrained candidate `tabicl_v2`: validation PR-AUC 0.272517 was 0.042503 below the selected candidate, outside the 0.01 near-tie margin.

At the fixed 0.5 threshold, both pretrained candidates produced all-negative validation predictions; their useful evidence here is ranking rather than thresholded case finding.

## Final frozen test result

Selected candidate: `tabpfn_v2`. Test evaluation was limited to the frozen selected pretrained candidate after evidence verification.

Test metrics: accuracy=0.950456; balanced_accuracy=0.500000; brier=0.042461; calibration_error=0.008821; f1=0.000000; n=767; negative=729; positive=38; positive_prevalence=0.049544; pr_auc=0.230704; precision=n/a; roc_auc=0.840120; sensitivity=0.000000; specificity=1.000000; threshold=0.500000.

## Calibration and class-imbalance interpretation

Calibration diagnostics are reported for score behavior only. A foundation-model score is not a calibrated acute stroke probability. Improved ranking, if observed, does not change the contextual-risk interpretation or establish clinical diagnostic performance.
The selected TabPFN v2 test score had PR-AUC 0.230704 and ROC-AUC 0.840120, but the fixed 0.5 threshold classified all 767 test rows as negative (sensitivity 0.000000); this is not a clinically validated operating threshold.

## Engineering trade-off

Measured local checkpoint size, initialization/context-fit cost, fixed-batch latency, whole-validation latency, and peak RSS are separated from upstream/documented model characteristics. Desktop measurements do not establish smartphone feasibility.

| Candidate | Checkpoint bytes | Init seconds | Context-fit seconds | Whole validation seconds | p50 sample ms | Peak RSS bytes |
|---|---:|---:|---:|---:|---:|---:|
| tabpfn_v2 | 29009539 | 1.066273300035391 | 19.29919280001195 | 68.13451760000316 | 55142.07379997242 | 694284288 |
| tabicl_v2 | 110368038 | 0.041841099970042706 | 59.88169140001992 | 30.926654600014444 | 26448.48030002322 | 521216000 |

## Final model-role recommendation

Reference/research metadata model: `tabpfn_v2` subject to the validation evidence and resource costs recorded here.
Lightweight edge/deployment metadata model: retain the canonical Logistic Regression alternative unless a separate production approval changes that role. No production integration is performed by this experiment.

## Limitations

- Severe class imbalance and only a small number of positive examples make precision, PR-AUC, calibration, and bootstrap intervals uncertain.
- The target and metadata branch are proxy/contextual risk evidence, not acute stroke diagnosis or a clinical probability.
- There is no paired multimodal clinical validation and no clinical probability interpretation.
- The historical Phase 4 baseline test metrics were previously exposed; the new trial uses them only as labeled context.
- Runtime and memory measurements are local desktop/process observations, not deployment feasibility claims.
