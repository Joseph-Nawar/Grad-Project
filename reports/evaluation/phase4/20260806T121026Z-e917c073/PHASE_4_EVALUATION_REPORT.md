# Phase 4 Evaluation Report

- Run ID: `20260806T121026Z-e917c073`
- Suite: `modality`
- Scope: reproducibility, proxy-partition metrics, deterministic robustness, and sensitivity analysis.
- Clinical diagnostic accuracy: not evaluated; no paired multimodal clinical dataset is available.

## Results

### Face

```json
{
  "n": 318,
  "direct": {
    "n": 318,
    "positive": 110,
    "negative": 208,
    "sensitivity": 0.9,
    "specificity": 0.9471153846153846,
    "precision": 0.9,
    "f1": 0.9,
    "roc_auc": 0.9817744755244755,
    "pr_auc": 0.9716443391899671,
    "brier": 0.06870462209334827,
    "calibration_error": 0.12706619818730033,
    "confusion_matrix": [
      [
        197,
        11
      ],
      [
        11,
        99
      ]
    ]
  },
  "adapter_coverage": {
    "accepted": 107,
    "rejected": 211,
    "quality_states": {
      "REJECT": 211,
      "PASS": 71,
      "WARN": 36
    },
    "rejection_reasons": {
      "pose_not_assessed": 211,
      "no_face": 191,
      "dimensions": 35,
      "face_too_small": 18
    }
  },
  "adapter_aware": {
    "n": 107,
    "positive": 29,
    "negative": 78,
    "sensitivity": 0.7931034482758621,
    "specificity": 0.9358974358974359,
    "precision": 0.8214285714285714,
    "f1": 0.8070175438596491,
    "roc_auc": 0.9606542882404951,
    "pr_auc": 0.9107901942657005,
    "brier": 0.0859720133868872,
    "calibration_error": 0.12301318067186903,
    "confusion_matrix": [
      [
        73,
        5
      ],
      [
        6,
        23
      ]
    ]
  },
  "bootstrap": {
    "metric": "roc_auc",
    "estimate": 0.9817744755244755,
    "lower": 0.9671263111888112,
    "upper": 0.9917832167832168,
    "iterations": 1000,
    "seed": 42,
    "unit": "sample-stratified"
  },
  "mean_inference_ms": 65.37078867924528,
  "artifacts": [
    "reports\\evaluation\\phase4\\20260806T121026Z-e917c073\\plots\\score_distribution.png"
  ]
}
```

### Speech

```json
{
  "n": 2849,
  "accepted": 2744,
  "failures": {
    "quality_rejected": 105
  },
  "quality_states": {
    "PASS": 2744,
    "REJECT": 105
  },
  "metrics": {
    "n": 2744,
    "positive": 879,
    "negative": 1865,
    "sensitivity": 0.9647326507394767,
    "specificity": 0.7276139410187668,
    "precision": 0.6253687315634219,
    "f1": 0.7588366890380314,
    "roc_auc": 0.9512442545300381,
    "pr_auc": 0.8993224002159479,
    "brier": 0.1243919622610949,
    "calibration_error": 0.1616217201166181,
    "confusion_matrix": [
      [
        1357,
        508
      ],
      [
        31,
        848
      ]
    ]
  },
  "speaker_bootstrap": {
    "metric": "roc_auc",
    "estimate": 0.9512442545300381,
    "lower": 0.8589854288264691,
    "upper": 0.9940171249521722,
    "iterations": 1000,
    "seed": 42,
    "unit": "group"
  }
}
```

### Metadata Context

```json
{
  "n": 767,
  "failures": {},
  "metrics": {
    "n": 767,
    "positive": 38,
    "negative": 729,
    "sensitivity": 0.8157894736842105,
    "specificity": 0.7187928669410151,
    "precision": 0.13135593220338984,
    "f1": 0.22627737226277372,
    "roc_auc": 0.8337665150530646,
    "pr_auc": 0.18565119693955895,
    "brier": 0.18132749525271655,
    "calibration_error": 0.28445717377951973,
    "confusion_matrix": [
      [
        524,
        205
      ],
      [
        7,
        31
      ]
    ]
  },
  "bootstrap": {
    "metric": "roc_auc",
    "estimate": 0.8337665150530646,
    "lower": 0.7744431809977619,
    "upper": 0.8909907226915024,
    "iterations": 1000,
    "seed": 42,
    "unit": "sample-stratified"
  }
}
```

## Interpretation

Scores are branch-specific proxy evidence or contextual-risk scores, not calibrated clinical stroke probabilities. Fusion outputs are engineering scenario results and must not be interpreted as clinical validation.
