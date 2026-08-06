# Phase 4 Evaluation Report

- Run ID: `20260806T131613Z-7df48bdd`
- Suite: `full`
- Scope: reproducibility, proxy-partition metrics, deterministic robustness, and sensitivity analysis.
- Direct model performance and runtime-adapter accepted-subset performance are reported separately.
- Rule verification, corruption robustness, missing-modality behavior, stability, and engineering sensitivity are not clinical accuracy measures.
- Clinical diagnostic accuracy and fusion clinical benefit: unsupported; no paired multimodal clinical dataset is available.

## Results

## Prominent coverage finding

The face adapter accepted 107 of 318 held-out images (33.6%) and rejected 211. Direct face-model metrics must not be read as runtime coverage metrics.

## Speech metric reconciliation

The direct canonical held-out evaluation produced ROC-AUC 0.9426 across 2849 samples. The runtime adapter accepted 2744 of 2849 samples and produced ROC-AUC 0.9512. The historical 0.9426 value matches the direct path (0.9426279894); 0.9512 is the accepted-subset adapter-aware result. The difference is therefore subset filtering by runtime quality rejection, not positive-class remapping or a changed model.

### Ablations

```json
{
  "scenarios": {
    "all_high": {
      "balanced_acute": {
        "band": "HIGH",
        "contributions": {
          "acute_symptoms": 0.24000000000000005,
          "face": 0.24000000000000005,
          "metadata_context": 0.08000000000000002,
          "speech": 0.24000000000000005
        },
        "evidence_score": 0.8000000000000003,
        "normalized_weights": {
          "acute_symptoms": 0.30000000000000004,
          "face": 0.30000000000000004,
          "metadata_context": 0.10000000000000002,
          "speech": 0.30000000000000004
        }
      },
      "canonical": {
        "band": "HIGH",
        "contributions": {
          "acute_symptoms": 0.20000000000000007,
          "face": 0.28,
          "metadata_context": 0.08000000000000002,
          "speech": 0.24000000000000005
        },
        "evidence_score": 0.8,
        "normalized_weights": {
          "acute_symptoms": 0.25000000000000006,
          "face": 0.35000000000000003,
          "metadata_context": 0.10000000000000002,
          "speech": 0.30000000000000004
        }
      },
      "symptom_emphasis": {
        "band": "HIGH",
        "contributions": {
          "acute_symptoms": 0.27999999999999997,
          "face": 0.24,
          "metadata_context": 0.08000000000000002,
          "speech": 0.2
        },
        "evidence_score": 0.8,
        "normalized_weights": {
          "acute_symptoms": 0.35,
          "face": 0.3,
          "metadata_context": 0.1,
          "speech": 0.25
        }
      }
    },
    "face_missing": {
      "balanced_acute": {
        "band": "HIGH",
        "contributions": {
          "acute_symptoms": 0.3428571428571429,
          "metadata_context": 0.11428571428571431,
          "speech": 0.3428571428571429
        },
        "evidence_score": 0.8000000000000002,
        "normalized_weights": {
          "acute_symptoms": 0.4285714285714286,
          "metadata_context": 0.14285714285714288,
          "speech": 0.4285714285714286
        }
      },
      "canonical": {
        "band": "HIGH",
        "contributions": {
          "acute_symptoms": 0.3076923076923077,
          "metadata_context": 0.12307692307692308,
          "speech": 0.36923076923076925
        },
        "evidence_score": 0.8,
        "normalized_weights": {
          "acute_symptoms": 0.3846153846153846,
          "metadata_context": 0.15384615384615385,
          "speech": 0.4615384615384615
        }
      },
      "symptom_emphasis": {
        "band": "HIGH",
        "contributions": {
          "acute_symptoms": 0.4,
          "metadata_context": 0.11428571428571431,
          "speech": 0.28571428571428575
        },
        "evidence_score": 0.8,
        "normalized_weights": {
          "acute_symptoms": 0.5,
          "metadata_context": 0.14285714285714288,
          "speech": 0.35714285714285715
        }
      }
    },
    "metadata_only": {
      "balanced_acute": {
        "band": "HIGH",
        "contributions": {
          "metadata_context": 0.8
        },
        "evidence_score": 0.8,
        "normalized_weights": {
          "metadata_context": 1.0
        }
      },
      "canonical": {
        "band": "HIGH",
        "contributions": {
          "metadata_context": 0.8
        },
        "evidence_score": 0.8,
        "normalized_weights": {
          "metadata_context": 1.0
        }
      },
      "symptom_emphasis": {
        "band": "HIGH",
        "contributions": {
          "metadata_context": 0.8
        },
        "evidence_score": 0.8,
        "normalized_weights": {
          "metadata_context": 1.0
        }
      }
    },
    "urgent": {
      "balanced_acute": {
        "band": "URGENT",
        "contributions": {
          "acute_symptoms": 0.28500000000000003,
          "face": 0.030000000000000006,
          "metadata_context": 0.010000000000000002,
          "speech": 0.030000000000000006
        },
        "evidence_score": 0.85,
        "normalized_weights": {
          "acute_symptoms": 0.30000000000000004,
          "face": 0.30000000000000004,
          "metadata_context": 0.10000000000000002,
          "speech": 0.30000000000000004
        }
      },
      "canonical": {
        "band": "URGENT",
        "contributions": {
          "acute_symptoms": 0.23750000000000004,
          "face": 0.035,
          "metadata_context": 0.010000000000000002,
          "speech": 0.030000000000000006
        },
        "evidence_score": 0.85,
        "normalized_weights": {
          "acute_symptoms": 0.25000000000000006,
          "face": 0.35000000000000003,
          "metadata_context": 0.10000000000000002,
          "speech": 0.30000000000000004
        }
      },
      "symptom_emphasis": {
        "band": "URGENT",
        "contributions": {
          "acute_symptoms": 0.33249999999999996,
          "face": 0.03,
          "metadata_context": 0.010000000000000002,
          "speech": 0.025
        },
        "evidence_score": 0.85,
        "normalized_weights": {
          "acute_symptoms": 0.35,
          "face": 0.3,
          "metadata_context": 0.1,
          "speech": 0.25
        }
      }
    }
  },
  "scope": "sensitivity analysis; no alternative is claimed more accurate"
}
```

### Cold Start

```json
{
  "runs": [
    {
      "assessment_ms": 3527.0262,
      "construction_ms": 19.7388,
      "exit_status": 0,
      "peak_child_rss_bytes": 736636928,
      "per_modality_ms": "{\"face\": 1537.3233, \"speech\": 1910.2073, \"metadata_context\": 5.8742, \"acute_symptoms\": 0.0265}",
      "process_wall_ms": 8907.249,
      "run": 1,
      "successful_fusion": true
    },
    {
      "assessment_ms": 3479.891,
      "construction_ms": 20.1154,
      "exit_status": 0,
      "peak_child_rss_bytes": 737734656,
      "per_modality_ms": "{\"face\": 1513.4365, \"speech\": 1892.2787, \"metadata_context\": 5.4652, \"acute_symptoms\": 0.0284}",
      "process_wall_ms": 8688.3025,
      "run": 2,
      "successful_fusion": true
    },
    {
      "assessment_ms": 3488.9107,
      "construction_ms": 19.8495,
      "exit_status": 0,
      "peak_child_rss_bytes": 732254208,
      "per_modality_ms": "{\"face\": 1557.1166, \"speech\": 1859.9767, \"metadata_context\": 5.6327, \"acute_symptoms\": 0.0284}",
      "process_wall_ms": 8586.5804,
      "run": 3,
      "successful_fusion": true
    }
  ],
  "successful_runs": 3
}
```

### Cold Start Subprocess Summary

```json
{
  "runs": [
    {
      "assessment_ms": 3527.0262,
      "construction_ms": 19.7388,
      "exit_status": 0,
      "peak_child_rss_bytes": 736636928,
      "per_modality_ms": "{\"face\": 1537.3233, \"speech\": 1910.2073, \"metadata_context\": 5.8742, \"acute_symptoms\": 0.0265}",
      "process_wall_ms": 8907.249,
      "run": 1,
      "successful_fusion": true
    },
    {
      "assessment_ms": 3479.891,
      "construction_ms": 20.1154,
      "exit_status": 0,
      "peak_child_rss_bytes": 737734656,
      "per_modality_ms": "{\"face\": 1513.4365, \"speech\": 1892.2787, \"metadata_context\": 5.4652, \"acute_symptoms\": 0.0284}",
      "process_wall_ms": 8688.3025,
      "run": 2,
      "successful_fusion": true
    },
    {
      "assessment_ms": 3488.9107,
      "construction_ms": 19.8495,
      "exit_status": 0,
      "peak_child_rss_bytes": 732254208,
      "per_modality_ms": "{\"face\": 1557.1166, \"speech\": 1859.9767, \"metadata_context\": 5.6327, \"acute_symptoms\": 0.0284}",
      "process_wall_ms": 8586.5804,
      "run": 3,
      "successful_fusion": true
    }
  ],
  "successful_runs": 3
}
```

### Corruption

```json
{
  "count": 10,
  "passed": 10,
  "results": [
    {
      "actual_status": "ok",
      "case": "empty_random_image",
      "expected": "isolated_or_schema_rejection",
      "failure_code": "decode_failed",
      "fusion_available": true,
      "pass": true,
      "usable_modalities": 3,
      "validation_or_isolation": "adapter_isolation_or_success"
    },
    {
      "actual_status": "ok",
      "case": "no_face_image",
      "expected": "isolated_or_schema_rejection",
      "failure_code": "pose_not_assessed;no_face",
      "fusion_available": true,
      "pass": true,
      "usable_modalities": 3,
      "validation_or_isolation": "adapter_isolation_or_success"
    },
    {
      "actual_status": "ok",
      "case": "very_small_image",
      "expected": "isolated_or_schema_rejection",
      "failure_code": "pose_not_assessed;dimensions;no_face",
      "fusion_available": true,
      "pass": true,
      "usable_modalities": 3,
      "validation_or_isolation": "adapter_isolation_or_success"
    },
    {
      "actual_status": "ok",
      "case": "blurred_image",
      "expected": "isolated_or_schema_rejection",
      "failure_code": "pose_not_assessed;no_face",
      "fusion_available": true,
      "pass": true,
      "usable_modalities": 3,
      "validation_or_isolation": "adapter_isolation_or_success"
    },
    {
      "actual_status": "ok",
      "case": "empty_random_audio",
      "expected": "isolated_or_schema_rejection",
      "failure_code": "decode_failed",
      "fusion_available": true,
      "pass": true,
      "usable_modalities": 3,
      "validation_or_isolation": "adapter_isolation_or_success"
    },
    {
      "actual_status": "ok",
      "case": "silent_audio",
      "expected": "isolated_or_schema_rejection",
      "failure_code": "low_energy",
      "fusion_available": true,
      "pass": true,
      "usable_modalities": 3,
      "validation_or_isolation": "adapter_isolation_or_success"
    },
    {
      "actual_status": "ok",
      "case": "short_audio",
      "expected": "isolated_or_schema_rejection",
      "failure_code": "too_short",
      "fusion_available": true,
      "pass": true,
      "usable_modalities": 3,
      "validation_or_isolation": "adapter_isolation_or_success"
    },
    {
      "actual_status": "ok",
      "case": "clipped_audio",
      "expected": "isolated_or_schema_rejection",
      "failure_code": "clipping",
      "fusion_available": true,
      "pass": true,
      "usable_modalities": 3,
      "validation_or_isolation": "adapter_isolation_or_success"
    },
    {
      "actual_status": "ok",
      "case": "missing_metadata",
      "expected": "isolated_or_schema_rejection",
      "failure_code": "",
      "fusion_available": true,
      "pass": true,
      "usable_modalities": 3,
      "validation_or_isolation": "adapter_isolation_or_success"
    },
    {
      "actual_status": "AssessmentInputError",
      "case": "invalid_symptoms",
      "expected": "isolated_or_schema_rejection",
      "failure_code": "AssessmentInputError",
      "fusion_available": false,
      "pass": true,
      "usable_modalities": 0,
      "validation_or_isolation": "schema_validation"
    }
  ]
}
```

### Face

```json
{
  "adapter_aware": {
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
    ],
    "f1": 0.8070175438596491,
    "n": 107,
    "negative": 78,
    "positive": 29,
    "pr_auc": 0.9107901942657005,
    "precision": 0.8214285714285714,
    "roc_auc": 0.9606542882404951,
    "sensitivity": 0.7931034482758621,
    "specificity": 0.9358974358974359
  },
  "adapter_coverage": {
    "accepted": 107,
    "coverage": 0.33647798742138363,
    "quality_states": {
      "PASS": 71,
      "REJECT": 211,
      "WARN": 36
    },
    "rejected": 211,
    "rejection_reasons": {
      "dimensions": 35,
      "face_too_small": 18,
      "no_face": 191,
      "pose_not_assessed": 211
    }
  },
  "artifacts": [
    "reports\\evaluation\\phase4\\final_complete\\plots\\face_score_distribution.png",
    "reports\\evaluation\\phase4\\final_complete\\plots\\face_roc.png",
    "reports\\evaluation\\phase4\\final_complete\\plots\\face_pr.png",
    "reports\\evaluation\\phase4\\final_complete\\plots\\face_calibration.png"
  ],
  "bootstrap": {
    "estimate": 0.9817744755244755,
    "iterations": 1000,
    "lower": 0.9671263111888112,
    "metric": "roc_auc",
    "seed": 42,
    "unit": "sample-stratified",
    "upper": 0.9917832167832168
  },
  "direct": {
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
    ],
    "f1": 0.9,
    "n": 318,
    "negative": 208,
    "positive": 110,
    "pr_auc": 0.9716443391899671,
    "precision": 0.9,
    "roc_auc": 0.9817744755244755,
    "sensitivity": 0.9,
    "specificity": 0.9471153846153846
  },
  "direct_positive_class": "Stroke",
  "mean_inference_ms": 63.014358805031456,
  "processed_direct": 318,
  "total_held_out": 318
}
```

### Face Adapter Coverage

```json
{
  "accepted": 107,
  "class_conditional_rejection_rate": {
    "NonStroke": 0.625,
    "Stroke": 0.7363636363636363
  },
  "coverage": 0.33647798742138363,
  "positive_class": "Stroke",
  "processed": 318,
  "rejected": 211,
  "total": 318
}
```

### Face Coverage

```json
{
  "accepted": 107,
  "coverage": 0.33647798742138363,
  "positive_class": "Stroke",
  "processed": 318,
  "rejected": 211,
  "total": 318
}
```

### Metadata Context

```json
{
  "artifacts": [
    "reports\\evaluation\\phase4\\final_complete\\plots\\metadata_score_distribution.png",
    "reports\\evaluation\\phase4\\final_complete\\plots\\metadata_roc.png",
    "reports\\evaluation\\phase4\\final_complete\\plots\\metadata_pr.png",
    "reports\\evaluation\\phase4\\final_complete\\plots\\metadata_calibration.png"
  ],
  "bootstrap": {
    "estimate": 0.8337665150530646,
    "iterations": 1000,
    "lower": 0.7744431809977619,
    "metric": "roc_auc",
    "seed": 42,
    "unit": "sample-stratified",
    "upper": 0.8909907226915024
  },
  "failures": {},
  "metrics": {
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
    ],
    "f1": 0.22627737226277372,
    "n": 767,
    "negative": 729,
    "positive": 38,
    "pr_auc": 0.18565119693955895,
    "precision": 0.13135593220338984,
    "roc_auc": 0.8337665150530646,
    "sensitivity": 0.8157894736842105,
    "specificity": 0.7187928669410151
  },
  "n": 767
}
```

### Missing Modalities

```json
{
  "combination_count": 16,
  "real_check_count": 4
}
```

### Profiling

```json
{
  "adapters": {
    "face": {
      "cold_runs": 3,
      "memory_increase_bytes": 0,
      "peak_rss_bytes": 826650624,
      "warm_median_ms": 79.1476,
      "warm_p95_ms": 81.3148,
      "warm_runs": 20
    },
    "metadata_context": {
      "cold_runs": 3,
      "memory_increase_bytes": 0,
      "peak_rss_bytes": 827133952,
      "warm_median_ms": 2.4146,
      "warm_p95_ms": 3.1053,
      "warm_runs": 20
    },
    "speech": {
      "cold_runs": 3,
      "memory_increase_bytes": 569344,
      "peak_rss_bytes": 827187200,
      "warm_median_ms": 37.84685,
      "warm_p95_ms": 39.8151,
      "warm_runs": 20
    }
  },
  "assessment_service": {
    "cold_runs": 3,
    "memory_increase_bytes": 50466816,
    "peak_rss_bytes": 826687488,
    "warm_median_ms": 121.7612,
    "warm_p95_ms": 124.56,
    "warm_runs": 20
  },
  "model_sizes_bytes": {
    "face": 9641144,
    "metadata_context": 6407,
    "speech": 29499167
  }
}
```

### Robustness

```json
{
  "count": 15,
  "no_usable_modalities": "insufficient_evidence",
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
  "scope": "engineering robustness and deterministic scenario consistency"
}
```

### Speech

```json
{
  "accepted": 2744,
  "adapter_aware": {
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
    ],
    "f1": 0.7588366890380314,
    "n": 2744,
    "negative": 1865,
    "positive": 879,
    "pr_auc": 0.8993224002159479,
    "precision": 0.6253687315634219,
    "roc_auc": 0.9512442545300381,
    "sensitivity": 0.9647326507394767,
    "specificity": 0.7276139410187668
  },
  "adapter_failures": {
    "quality_rejected": 105
  },
  "artifacts": [
    "reports\\evaluation\\phase4\\final_complete\\plots\\speech_score_distribution.png",
    "reports\\evaluation\\phase4\\final_complete\\plots\\speech_roc.png",
    "reports\\evaluation\\phase4\\final_complete\\plots\\speech_pr.png",
    "reports\\evaluation\\phase4\\final_complete\\plots\\speech_calibration.png"
  ],
  "coverage": 0.9631449631449631,
  "direct": {
    "brier": 0.12458719238719239,
    "calibration_error": 0.14445536445536447,
    "confusion_matrix": [
      [
        1359,
        508
      ],
      [
        52,
        930
      ]
    ],
    "f1": 0.768595041322314,
    "n": 2849,
    "negative": 1867,
    "positive": 982,
    "pr_auc": 0.8932653414911541,
    "precision": 0.6467315716272601,
    "roc_auc": 0.9426279894010781,
    "sensitivity": 0.9470468431771895,
    "specificity": 0.727905731119443
  },
  "direct_failures": {},
  "processed_direct": 2849,
  "quality_states": {
    "PASS": 2744,
    "REJECT": 105
  },
  "rejected": 105,
  "speaker_bootstrap": {
    "estimate": 0.9512442545300381,
    "iterations": 1000,
    "lower": 0.8589854288264691,
    "metric": "roc_auc",
    "seed": 42,
    "unit": "group",
    "upper": 0.9940171249521722
  },
  "total_held_out": 2849
}
```

### Speech Speakers

```json
{
  "accepted": 2744,
  "speaker_count": 9
}
```

### Speech Speaker Summary

```json
{
  "single_class_per_speaker_roc_auc": "not calculated",
  "speaker_count": 9,
  "speakers": [
    {
      "accepted_count": 207,
      "accuracy": 0.14492753623188406,
      "class": "control",
      "correct_count": 30,
      "coverage": 1.0,
      "max_score": 0.94,
      "mean_score": 0.6730756843800323,
      "median_score": 0.6833333333333333,
      "min_score": 0.2833333333333333,
      "quality_failures": 0,
      "sample_count": 207,
      "speaker_id": "wav_headMic_FC03S03"
    },
    {
      "accepted_count": 360,
      "accuracy": 0.25833333333333336,
      "class": "control",
      "correct_count": 93,
      "coverage": 1.0,
      "max_score": 0.88,
      "mean_score": 0.5764722222222223,
      "median_score": 0.5766666666666667,
      "min_score": 0.2,
      "quality_failures": 0,
      "sample_count": 360,
      "speaker_id": "wav_headMic_MC01S02"
    },
    {
      "accepted_count": 134,
      "accuracy": 0.8805970149253731,
      "class": "dysarthric",
      "correct_count": 118,
      "coverage": 1.0,
      "max_score": 0.97,
      "mean_score": 0.7089054726368159,
      "median_score": 0.7483333333333333,
      "min_score": 0.15333333333333332,
      "quality_failures": 0,
      "sample_count": 134,
      "speaker_id": "wav_arrayMic_F01"
    },
    {
      "accepted_count": 299,
      "accuracy": 0.9063545150501672,
      "class": "control",
      "correct_count": 271,
      "coverage": 0.9966666666666667,
      "max_score": 0.8933333333333333,
      "mean_score": 0.25319955406911926,
      "median_score": 0.21666666666666667,
      "min_score": 0.016666666666666666,
      "quality_failures": 1,
      "sample_count": 300,
      "speaker_id": "wav_arrayMic_MC03S02"
    },
    {
      "accepted_count": 400,
      "accuracy": 0.9125,
      "class": "control",
      "correct_count": 365,
      "coverage": 1.0,
      "max_score": 0.7566666666666667,
      "mean_score": 0.2690416666666667,
      "median_score": 0.2516666666666667,
      "min_score": 0.02,
      "quality_failures": 0,
      "sample_count": 400,
      "speaker_id": "wav_arrayMic_FC03S01"
    },
    {
      "accepted_count": 100,
      "accuracy": 0.93,
      "class": "dysarthric",
      "correct_count": 93,
      "coverage": 0.591715976331361,
      "max_score": 0.9566666666666667,
      "mean_score": 0.7515666666666667,
      "median_score": 0.7933333333333333,
      "min_score": 0.26,
      "quality_failures": 69,
      "sample_count": 169,
      "speaker_id": "wav_headMic_M02S02"
    },
    {
      "accepted_count": 359,
      "accuracy": 0.9860724233983287,
      "class": "dysarthric",
      "correct_count": 354,
      "coverage": 0.9134860050890585,
      "max_score": 0.9833333333333333,
      "mean_score": 0.8305663881151348,
      "median_score": 0.8633333333333333,
      "min_score": 0.07333333333333333,
      "quality_failures": 34,
      "sample_count": 393,
      "speaker_id": "wav_headMic_M05S02"
    },
    {
      "accepted_count": 286,
      "accuracy": 0.9895104895104895,
      "class": "dysarthric",
      "correct_count": 283,
      "coverage": 1.0,
      "max_score": 0.9966666666666667,
      "mean_score": 0.8700000000000001,
      "median_score": 0.8916666666666666,
      "min_score": 0.2833333333333333,
      "quality_failures": 0,
      "sample_count": 286,
      "speaker_id": "wav_headMic_M01S02"
    },
    {
      "accepted_count": 599,
      "accuracy": 0.998330550918197,
      "class": "control",
      "correct_count": 598,
      "coverage": 0.9983333333333333,
      "max_score": 0.58,
      "mean_score": 0.09149693934335001,
      "median_score": 0.08,
      "min_score": 0.0,
      "quality_failures": 1,
      "sample_count": 600,
      "speaker_id": "wav_headMic_MC03S01"
    }
  ]
}
```

### Stability

```json
{
  "acute_symptoms": {
    "band_consistent": true,
    "label_consistent": true,
    "max_absolute_difference": 0.0,
    "quality_consistent": true,
    "runs": 20,
    "score_max": 0.35,
    "score_min": 0.35
  },
  "assessment_service": {
    "band_consistent": true,
    "label_consistent": true,
    "max_absolute_difference": 0.0,
    "quality_consistent": true,
    "runs": 20,
    "score_max": 0.5072364544197587,
    "score_min": 0.5072364544197587
  },
  "face": {
    "band_consistent": true,
    "label_consistent": true,
    "max_absolute_difference": 0.0,
    "quality_consistent": true,
    "runs": 20,
    "score_max": 0.1502818465232849,
    "score_min": 0.1502818465232849
  },
  "metadata_context": {
    "band_consistent": true,
    "label_consistent": true,
    "max_absolute_difference": 0.0,
    "quality_consistent": true,
    "runs": 20,
    "score_max": 0.9013780813660891,
    "score_min": 0.9013780813660891
  },
  "speech": {
    "band_consistent": true,
    "label_consistent": true,
    "max_absolute_difference": 0.0,
    "quality_consistent": true,
    "runs": 20,
    "score_max": 0.9233333333333333,
    "score_min": 0.9233333333333333
  }
}
```

### Stability Results

```json
{
  "acute_symptoms": {
    "band_consistent": true,
    "label_consistent": true,
    "max_absolute_difference": 0.0,
    "quality_consistent": true,
    "runs": 20,
    "score_max": 0.35,
    "score_min": 0.35
  },
  "assessment_service": {
    "band_consistent": true,
    "label_consistent": true,
    "max_absolute_difference": 0.0,
    "quality_consistent": true,
    "runs": 20,
    "score_max": 0.5072364544197587,
    "score_min": 0.5072364544197587
  },
  "face": {
    "band_consistent": true,
    "label_consistent": true,
    "max_absolute_difference": 0.0,
    "quality_consistent": true,
    "runs": 20,
    "score_max": 0.1502818465232849,
    "score_min": 0.1502818465232849
  },
  "metadata_context": {
    "band_consistent": true,
    "label_consistent": true,
    "max_absolute_difference": 0.0,
    "quality_consistent": true,
    "runs": 20,
    "score_max": 0.9013780813660891,
    "score_min": 0.9013780813660891
  },
  "speech": {
    "band_consistent": true,
    "label_consistent": true,
    "max_absolute_difference": 0.0,
    "quality_consistent": true,
    "runs": 20,
    "score_max": 0.9233333333333333,
    "score_min": 0.9233333333333333
  }
}
```

### Symptoms

```json
{
  "rule_verification_only": true,
  "scenario_count": 7,
  "scenarios": {
    "all_fast": {
      "hard_escalation": true,
      "label": "URGENT",
      "score": 0.85,
      "warnings": [
        "Multiple FAST warning signs reported. Treat as urgent screening concern.",
        "Reported onset is within a commonly referenced emergency treatment window.",
        "This module is a research screening component and is not a clinical diagnosis."
      ]
    },
    "face": {
      "hard_escalation": false,
      "label": "MODERATE",
      "score": 0.35,
      "warnings": [
        "Symptom onset time is unknown.",
        "This module is a research screening component and is not a clinical diagnosis."
      ]
    },
    "none": {
      "hard_escalation": false,
      "label": "LOW",
      "score": 0.0,
      "warnings": [
        "Symptom onset time is unknown.",
        "This module is a research screening component and is not a clinical diagnosis."
      ]
    },
    "resolved": {
      "hard_escalation": false,
      "label": "MODERATE",
      "score": 0.35,
      "warnings": [
        "Reported onset is within a commonly referenced emergency treatment window.",
        "Symptoms are reported as resolved. Transient symptoms may still require urgent assessment.",
        "This module is a research screening component and is not a clinical diagnosis."
      ]
    },
    "speech": {
      "hard_escalation": false,
      "label": "MODERATE",
      "score": 0.35,
      "warnings": [
        "Symptom onset time is unknown.",
        "This module is a research screening component and is not a clinical diagnosis."
      ]
    },
    "two_core": {
      "hard_escalation": true,
      "label": "URGENT",
      "score": 0.85,
      "warnings": [
        "Multiple FAST warning signs reported. Treat as urgent screening concern.",
        "Reported onset is within a commonly referenced emergency treatment window.",
        "This module is a research screening component and is not a clinical diagnosis."
      ]
    },
    "unknown_onset": {
      "hard_escalation": false,
      "label": "MODERATE",
      "score": 0.35,
      "warnings": [
        "Symptom onset time is unknown.",
        "This module is a research screening component and is not a clinical diagnosis."
      ]
    }
  }
}
```

### System Corruption Summary

```json
{
  "count": 10,
  "passed": 10,
  "results": [
    {
      "actual_status": "ok",
      "case": "empty_random_image",
      "expected": "isolated_or_schema_rejection",
      "failure_code": "decode_failed",
      "fusion_available": true,
      "pass": true,
      "usable_modalities": 3,
      "validation_or_isolation": "adapter_isolation_or_success"
    },
    {
      "actual_status": "ok",
      "case": "no_face_image",
      "expected": "isolated_or_schema_rejection",
      "failure_code": "pose_not_assessed;no_face",
      "fusion_available": true,
      "pass": true,
      "usable_modalities": 3,
      "validation_or_isolation": "adapter_isolation_or_success"
    },
    {
      "actual_status": "ok",
      "case": "very_small_image",
      "expected": "isolated_or_schema_rejection",
      "failure_code": "pose_not_assessed;dimensions;no_face",
      "fusion_available": true,
      "pass": true,
      "usable_modalities": 3,
      "validation_or_isolation": "adapter_isolation_or_success"
    },
    {
      "actual_status": "ok",
      "case": "blurred_image",
      "expected": "isolated_or_schema_rejection",
      "failure_code": "pose_not_assessed;no_face",
      "fusion_available": true,
      "pass": true,
      "usable_modalities": 3,
      "validation_or_isolation": "adapter_isolation_or_success"
    },
    {
      "actual_status": "ok",
      "case": "empty_random_audio",
      "expected": "isolated_or_schema_rejection",
      "failure_code": "decode_failed",
      "fusion_available": true,
      "pass": true,
      "usable_modalities": 3,
      "validation_or_isolation": "adapter_isolation_or_success"
    },
    {
      "actual_status": "ok",
      "case": "silent_audio",
      "expected": "isolated_or_schema_rejection",
      "failure_code": "low_energy",
      "fusion_available": true,
      "pass": true,
      "usable_modalities": 3,
      "validation_or_isolation": "adapter_isolation_or_success"
    },
    {
      "actual_status": "ok",
      "case": "short_audio",
      "expected": "isolated_or_schema_rejection",
      "failure_code": "too_short",
      "fusion_available": true,
      "pass": true,
      "usable_modalities": 3,
      "validation_or_isolation": "adapter_isolation_or_success"
    },
    {
      "actual_status": "ok",
      "case": "clipped_audio",
      "expected": "isolated_or_schema_rejection",
      "failure_code": "clipping",
      "fusion_available": true,
      "pass": true,
      "usable_modalities": 3,
      "validation_or_isolation": "adapter_isolation_or_success"
    },
    {
      "actual_status": "ok",
      "case": "missing_metadata",
      "expected": "isolated_or_schema_rejection",
      "failure_code": "",
      "fusion_available": true,
      "pass": true,
      "usable_modalities": 3,
      "validation_or_isolation": "adapter_isolation_or_success"
    },
    {
      "actual_status": "AssessmentInputError",
      "case": "invalid_symptoms",
      "expected": "isolated_or_schema_rejection",
      "failure_code": "AssessmentInputError",
      "fusion_available": false,
      "pass": true,
      "usable_modalities": 0,
      "validation_or_isolation": "schema_validation"
    }
  ]
}
```

## Warnings

- Metadata artifact was serialized with scikit-learn 1.4.2 and loaded in the validated 1.5.2 runtime.

## Interpretation

Scores are branch-specific proxy evidence or contextual-risk scores, not calibrated clinical stroke probabilities. Fusion outputs are engineering scenario results and must not be interpreted as clinical validation.
