# Phase 4 Evaluation Report

- Run ID: `20260806T132724Z-f7e51c5d`
- Suite: `smoke`
- Scope: reproducibility, proxy-partition metrics, deterministic robustness, and sensitivity analysis.
- Direct model performance and runtime-adapter accepted-subset performance are reported separately.
- Rule verification, corruption robustness, missing-modality behavior, stability, and engineering sensitivity are not clinical accuracy measures.
- Clinical diagnostic accuracy and fusion clinical benefit: unsupported; no paired multimodal clinical dataset is available.

## Results

### Face

```json
{
  "total_held_out": 5,
  "processed_direct": 5,
  "direct_positive_class": "Stroke",
  "direct": {
    "n": 5,
    "positive": 0,
    "negative": 5,
    "sensitivity": null,
    "specificity": 0.6,
    "precision": 0.0,
    "f1": 0.0,
    "roc_auc": null,
    "pr_auc": null,
    "brier": 0.16252718409224987,
    "calibration_error": 0.288933839276433,
    "confusion_matrix": [
      [
        3,
        2
      ],
      [
        0,
        0
      ]
    ]
  },
  "adapter_coverage": {
    "accepted": 2,
    "rejected": 3,
    "coverage": 0.4,
    "quality_states": {
      "REJECT": 3,
      "WARN": 2
    },
    "rejection_reasons": {
      "pose_not_assessed": 3,
      "no_face": 2,
      "dimensions": 1,
      "face_too_small": 1
    }
  },
  "adapter_aware": {
    "n": 2,
    "positive": 0,
    "negative": 2,
    "sensitivity": null,
    "specificity": 1.0,
    "precision": null,
    "f1": null,
    "roc_auc": null,
    "pr_auc": null,
    "brier": 0.01228156845365131,
    "calibration_error": 0.09738110937178135,
    "confusion_matrix": [
      [
        2,
        0
      ],
      [
        0,
        0
      ]
    ]
  },
  "bootstrap": {
    "metric": "roc_auc",
    "estimate": null,
    "lower": null,
    "upper": null,
    "iterations": 25,
    "seed": 42,
    "unit": "sample-stratified"
  },
  "mean_inference_ms": 190.14425999999997,
  "artifacts": [
    "reports\\evaluation\\phase4\\20260806T132724Z-f7e51c5d\\plots\\face_score_distribution.png",
    "reports\\evaluation\\phase4\\20260806T132724Z-f7e51c5d\\plots\\face_roc.png",
    "reports\\evaluation\\phase4\\20260806T132724Z-f7e51c5d\\plots\\face_pr.png",
    "reports\\evaluation\\phase4\\20260806T132724Z-f7e51c5d\\plots\\face_calibration.png"
  ]
}
```

### Speech

```json
{
  "total_held_out": 5,
  "processed_direct": 5,
  "direct_failures": {},
  "direct": {
    "n": 5,
    "positive": 5,
    "negative": 0,
    "sensitivity": 1.0,
    "specificity": null,
    "precision": 1.0,
    "f1": 1.0,
    "roc_auc": null,
    "pr_auc": null,
    "brier": 0.015348888888888884,
    "calibration_error": 0.1166666666666666,
    "confusion_matrix": [
      [
        0,
        0
      ],
      [
        0,
        5
      ]
    ]
  },
  "accepted": 5,
  "rejected": 0,
  "coverage": 1.0,
  "adapter_failures": {},
  "quality_states": {
    "PASS": 5
  },
  "adapter_aware": {
    "n": 5,
    "positive": 5,
    "negative": 0,
    "sensitivity": 1.0,
    "specificity": null,
    "precision": 1.0,
    "f1": 1.0,
    "roc_auc": null,
    "pr_auc": null,
    "brier": 0.015348888888888884,
    "calibration_error": 0.1166666666666666,
    "confusion_matrix": [
      [
        0,
        0
      ],
      [
        0,
        5
      ]
    ]
  },
  "artifacts": [
    "reports\\evaluation\\phase4\\20260806T132724Z-f7e51c5d\\plots\\speech_score_distribution.png",
    "reports\\evaluation\\phase4\\20260806T132724Z-f7e51c5d\\plots\\speech_roc.png",
    "reports\\evaluation\\phase4\\20260806T132724Z-f7e51c5d\\plots\\speech_pr.png",
    "reports\\evaluation\\phase4\\20260806T132724Z-f7e51c5d\\plots\\speech_calibration.png"
  ],
  "speaker_bootstrap": {
    "metric": "roc_auc",
    "estimate": null,
    "lower": null,
    "upper": null,
    "iterations": 25,
    "seed": 42,
    "unit": "group"
  }
}
```

### Metadata Context

```json
{
  "n": 5,
  "failures": {},
  "metrics": {
    "n": 5,
    "positive": 5,
    "negative": 0,
    "sensitivity": 0.8,
    "specificity": null,
    "precision": 1.0,
    "f1": 0.8888888888888888,
    "roc_auc": null,
    "pr_auc": null,
    "brier": 0.16916632051248753,
    "calibration_error": 0.2978539409259218,
    "confusion_matrix": [
      [
        0,
        0
      ],
      [
        1,
        4
      ]
    ]
  },
  "artifacts": [
    "reports\\evaluation\\phase4\\20260806T132724Z-f7e51c5d\\plots\\metadata_score_distribution.png",
    "reports\\evaluation\\phase4\\20260806T132724Z-f7e51c5d\\plots\\metadata_roc.png",
    "reports\\evaluation\\phase4\\20260806T132724Z-f7e51c5d\\plots\\metadata_pr.png",
    "reports\\evaluation\\phase4\\20260806T132724Z-f7e51c5d\\plots\\metadata_calibration.png"
  ],
  "bootstrap": {
    "metric": "roc_auc",
    "estimate": null,
    "lower": null,
    "upper": null,
    "iterations": 25,
    "seed": 42,
    "unit": "sample-stratified"
  }
}
```

### Symptoms

```json
{
  "scenario_count": 7,
  "scenarios": {
    "none": {
      "score": 0.0,
      "label": "LOW",
      "hard_escalation": false,
      "warnings": [
        "Symptom onset time is unknown.",
        "This module is a research screening component and is not a clinical diagnosis."
      ]
    },
    "face": {
      "score": 0.35,
      "label": "MODERATE",
      "hard_escalation": false,
      "warnings": [
        "Symptom onset time is unknown.",
        "This module is a research screening component and is not a clinical diagnosis."
      ]
    },
    "speech": {
      "score": 0.35,
      "label": "MODERATE",
      "hard_escalation": false,
      "warnings": [
        "Symptom onset time is unknown.",
        "This module is a research screening component and is not a clinical diagnosis."
      ]
    },
    "two_core": {
      "score": 0.85,
      "label": "URGENT",
      "hard_escalation": true,
      "warnings": [
        "Multiple FAST warning signs reported. Treat as urgent screening concern.",
        "Reported onset is within a commonly referenced emergency treatment window.",
        "This module is a research screening component and is not a clinical diagnosis."
      ]
    },
    "resolved": {
      "score": 0.35,
      "label": "MODERATE",
      "hard_escalation": false,
      "warnings": [
        "Reported onset is within a commonly referenced emergency treatment window.",
        "Symptoms are reported as resolved. Transient symptoms may still require urgent assessment.",
        "This module is a research screening component and is not a clinical diagnosis."
      ]
    },
    "unknown_onset": {
      "score": 0.35,
      "label": "MODERATE",
      "hard_escalation": false,
      "warnings": [
        "Symptom onset time is unknown.",
        "This module is a research screening component and is not a clinical diagnosis."
      ]
    },
    "all_fast": {
      "score": 0.85,
      "label": "URGENT",
      "hard_escalation": true,
      "warnings": [
        "Multiple FAST warning signs reported. Treat as urgent screening concern.",
        "Reported onset is within a commonly referenced emergency treatment window.",
        "This module is a research screening component and is not a clinical diagnosis."
      ]
    }
  },
  "rule_verification_only": true
}
```

### Robustness

```json
{
  "non_empty_combinations": [
    [
      "face"
    ],
    [
      "speech"
    ],
    [
      "metadata_context"
    ],
    [
      "acute_symptoms"
    ],
    [
      "face",
      "speech"
    ],
    [
      "face",
      "metadata_context"
    ],
    [
      "face",
      "acute_symptoms"
    ],
    [
      "speech",
      "metadata_context"
    ],
    [
      "speech",
      "acute_symptoms"
    ],
    [
      "metadata_context",
      "acute_symptoms"
    ],
    [
      "face",
      "speech",
      "metadata_context"
    ],
    [
      "face",
      "speech",
      "acute_symptoms"
    ],
    [
      "face",
      "metadata_context",
      "acute_symptoms"
    ],
    [
      "speech",
      "metadata_context",
      "acute_symptoms"
    ],
    [
      "face",
      "speech",
      "metadata_context",
      "acute_symptoms"
    ]
  ],
  "count": 15,
  "no_usable_modalities": "insufficient_evidence",
  "scope": "engineering robustness and deterministic scenario consistency"
}
```

### Ablations

```json
{
  "scope": "sensitivity analysis; no alternative is claimed more accurate",
  "scenarios": {
    "all_high": {
      "canonical": {
        "evidence_score": 0.8,
        "band": "HIGH",
        "normalized_weights": {
          "face": 0.35000000000000003,
          "speech": 0.30000000000000004,
          "acute_symptoms": 0.25000000000000006,
          "metadata_context": 0.10000000000000002
        },
        "contributions": {
          "face": 0.28,
          "speech": 0.24000000000000005,
          "acute_symptoms": 0.20000000000000007,
          "metadata_context": 0.08000000000000002
        }
      },
      "balanced_acute": {
        "evidence_score": 0.8000000000000003,
        "band": "HIGH",
        "normalized_weights": {
          "face": 0.30000000000000004,
          "speech": 0.30000000000000004,
          "acute_symptoms": 0.30000000000000004,
          "metadata_context": 0.10000000000000002
        },
        "contributions": {
          "face": 0.24000000000000005,
          "speech": 0.24000000000000005,
          "acute_symptoms": 0.24000000000000005,
          "metadata_context": 0.08000000000000002
        }
      },
      "symptom_emphasis": {
        "evidence_score": 0.8,
        "band": "HIGH",
        "normalized_weights": {
          "face": 0.3,
          "speech": 0.25,
          "acute_symptoms": 0.35,
          "metadata_context": 0.1
        },
        "contributions": {
          "face": 0.24,
          "speech": 0.2,
          "acute_symptoms": 0.27999999999999997,
          "metadata_context": 0.08000000000000002
        }
      }
    },
    "face_missing": {
      "canonical": {
        "evidence_score": 0.8,
        "band": "HIGH",
        "normalized_weights": {
          "speech": 0.4615384615384615,
          "acute_symptoms": 0.3846153846153846,
          "metadata_context": 0.15384615384615385
        },
        "contributions": {
          "speech": 0.36923076923076925,
          "acute_symptoms": 0.3076923076923077,
          "metadata_context": 0.12307692307692308
        }
      },
      "balanced_acute": {
        "evidence_score": 0.8000000000000002,
        "band": "HIGH",
        "normalized_weights": {
          "speech": 0.4285714285714286,
          "acute_symptoms": 0.4285714285714286,
          "metadata_context": 0.14285714285714288
        },
        "contributions": {
          "speech": 0.3428571428571429,
          "acute_symptoms": 0.3428571428571429,
          "metadata_context": 0.11428571428571431
        }
      },
      "symptom_emphasis": {
        "evidence_score": 0.8,
        "band": "HIGH",
        "normalized_weights": {
          "speech": 0.35714285714285715,
          "acute_symptoms": 0.5,
          "metadata_context": 0.14285714285714288
        },
        "contributions": {
          "speech": 0.28571428571428575,
          "acute_symptoms": 0.4,
          "metadata_context": 0.11428571428571431
        }
      }
    },
    "metadata_only": {
      "canonical": {
        "evidence_score": 0.8,
        "band": "HIGH",
        "normalized_weights": {
          "metadata_context": 1.0
        },
        "contributions": {
          "metadata_context": 0.8
        }
      },
      "balanced_acute": {
        "evidence_score": 0.8,
        "band": "HIGH",
        "normalized_weights": {
          "metadata_context": 1.0
        },
        "contributions": {
          "metadata_context": 0.8
        }
      },
      "symptom_emphasis": {
        "evidence_score": 0.8,
        "band": "HIGH",
        "normalized_weights": {
          "metadata_context": 1.0
        },
        "contributions": {
          "metadata_context": 0.8
        }
      }
    },
    "urgent": {
      "canonical": {
        "evidence_score": 0.85,
        "band": "URGENT",
        "normalized_weights": {
          "face": 0.35000000000000003,
          "speech": 0.30000000000000004,
          "acute_symptoms": 0.25000000000000006,
          "metadata_context": 0.10000000000000002
        },
        "contributions": {
          "face": 0.035,
          "speech": 0.030000000000000006,
          "acute_symptoms": 0.23750000000000004,
          "metadata_context": 0.010000000000000002
        }
      },
      "balanced_acute": {
        "evidence_score": 0.85,
        "band": "URGENT",
        "normalized_weights": {
          "face": 0.30000000000000004,
          "speech": 0.30000000000000004,
          "acute_symptoms": 0.30000000000000004,
          "metadata_context": 0.10000000000000002
        },
        "contributions": {
          "face": 0.030000000000000006,
          "speech": 0.030000000000000006,
          "acute_symptoms": 0.28500000000000003,
          "metadata_context": 0.010000000000000002
        }
      },
      "symptom_emphasis": {
        "evidence_score": 0.85,
        "band": "URGENT",
        "normalized_weights": {
          "face": 0.3,
          "speech": 0.25,
          "acute_symptoms": 0.35,
          "metadata_context": 0.1
        },
        "contributions": {
          "face": 0.03,
          "speech": 0.025,
          "acute_symptoms": 0.33249999999999996,
          "metadata_context": 0.010000000000000002
        }
      }
    }
  }
}
```

## Interpretation

Scores are branch-specific proxy evidence or contextual-risk scores, not calibrated clinical stroke probabilities. Fusion outputs are engineering scenario results and must not be interpreted as clinical validation.
