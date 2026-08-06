# Phase 4 Evaluation Report

- Run ID: `20260806T122233Z-ec063a2a`
- Suite: `full`
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
  "mean_inference_ms": 63.12666886792453,
  "artifacts": [
    "reports\\evaluation\\phase4\\final\\plots\\face_score_distribution.png",
    "reports\\evaluation\\phase4\\final\\plots\\face_roc.png",
    "reports\\evaluation\\phase4\\final\\plots\\face_pr.png",
    "reports\\evaluation\\phase4\\final\\plots\\face_calibration.png"
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
  "artifacts": [
    "reports\\evaluation\\phase4\\final\\plots\\speech_score_distribution.png",
    "reports\\evaluation\\phase4\\final\\plots\\speech_roc.png",
    "reports\\evaluation\\phase4\\final\\plots\\speech_pr.png",
    "reports\\evaluation\\phase4\\final\\plots\\speech_calibration.png"
  ],
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
  "artifacts": [
    "reports\\evaluation\\phase4\\final\\plots\\metadata_score_distribution.png",
    "reports\\evaluation\\phase4\\final\\plots\\metadata_roc.png",
    "reports\\evaluation\\phase4\\final\\plots\\metadata_pr.png",
    "reports\\evaluation\\phase4\\final\\plots\\metadata_calibration.png"
  ],
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

### Profiling

```json
{
  "assessment_service": {
    "cold_runs": 3,
    "warm_runs": 20,
    "warm_median_ms": 120.5898,
    "warm_p95_ms": 133.2414,
    "peak_rss_bytes": 794652672,
    "memory_increase_bytes": 0
  },
  "adapters": {
    "face": {
      "cold_runs": 3,
      "warm_runs": 20,
      "warm_median_ms": 76.80535,
      "warm_p95_ms": 81.0086,
      "peak_rss_bytes": 788045824,
      "memory_increase_bytes": 0
    },
    "speech": {
      "cold_runs": 3,
      "warm_runs": 20,
      "warm_median_ms": 37.897149999999996,
      "warm_p95_ms": 38.6656,
      "peak_rss_bytes": 788869120,
      "memory_increase_bytes": 1114112
    },
    "metadata_context": {
      "cold_runs": 3,
      "warm_runs": 20,
      "warm_median_ms": 2.34575,
      "warm_p95_ms": 2.7037,
      "peak_rss_bytes": 788824064,
      "memory_increase_bytes": 0
    }
  },
  "model_sizes_bytes": {
    "face": 9641144,
    "speech": 29499167,
    "metadata_context": 6407
  }
}
```

## Interpretation

Scores are branch-specific proxy evidence or contextual-risk scores, not calibrated clinical stroke probabilities. Fusion outputs are engineering scenario results and must not be interpreted as clinical validation.
