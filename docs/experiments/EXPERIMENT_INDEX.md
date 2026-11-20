# Experiment and metric provenance index

This index keeps the accepted MIUCC 2026 paper, the current repository evaluation, and the deployment-oriented optimized profile distinct. No paper PDF or authoritative paper-specific metric record is checked in, so paper methods and numbers are not inferred here.

| Evidence set | Profile / context | Authoritative files | Safe interpretation |
| --- | --- | --- | --- |
| Earlier MIUCC 2026 research paper | Earlier framework version; owner-confirmed accepted and presented | No accepted manuscript or official proceedings record is present in this repository. The current `README.md` records acceptance for presentation. | Do not attribute current repository metrics to the paper. Do not claim a DOI or IEEE Xplore indexing. |
| Phase 4 held-out proxy evaluation | Current canonical branch models, seed 42, Python 3.11.9, Windows 10, 1,000 bootstrap iterations; run `20260806T131613Z-7df48bdd` | `reports/evaluation/phase4/final_complete/run_manifest.json`, `PHASE_4_EVALUATION_REPORT.md`, `face.json`, `speech.json`, `metadata_context.json`, `symptoms.json`, and `missing_modalities.json` | Branch-specific proxy tasks and contract checks only. The result is not end-to-end clinical performance. Keep dataset/protocol limitations from the associated model and dataset cards. |
| Stage 5 optimized edge comparison | `mixed-litert` selected bundle versus baseline; 10 warm runs, Windows development environment | `reports/production/stage5/stage5_summary.json`, `bundle_benchmarks.json`, `conversion_environment.json`, and `docker_closure.json` | Historical committed runtime/bundle evidence. The 23,421,513-byte artifact bundle and warm assessment p50/p95 are not total application size, end-to-end HTTP latency, or native smartphone performance. |
| Current optimized runtime profile | LiteRT MobileNetV2 face branch; ONNX MFCC + Random Forest speech branch; scikit-learn Logistic Regression contextual metadata; same deterministic symptom rules | `config/edge_runtime_registry.json`, `rural_stroke_assist/assessment/factory.py`, `rural_stroke_assist/inference/`, `reports/production/stage5/` | Describes the deployment-oriented runtime profile, not the earlier paper. A current UI capture must name the profile used and avoid suggesting clinical validation. |
| Research-reference profile | ImageNet MobileNetV2; frozen DistilHuBERT representation plus logistic regression; TabPFN v2; same deterministic symptoms | `config/pretrained_reference_registry.json`, `rural_stroke_assist/inference/reference_runtime.py` | A distinct orchestration profile. Its presence does not establish that it matches the accepted paper. |

## Current-code result index

| Result | Provenance | Limitation that must accompany any use |
| --- | --- | --- |
| Face direct ROC-AUC 0.9818 and PR-AUC 0.9716, `n=318` | Phase 4 `face.json`, direct path | Public proxy-image label task. Runtime quality gate accepted only 107/318 (33.65%). Not facial-droop sensitivity or clinical accuracy. |
| Face runtime quality gate 107/318 accepted | Phase 4 `face.json` and adapter coverage artifacts | Low, uneven coverage; direct and accepted-subset metrics apply to different samples. |
| Speech direct ROC-AUC 0.9426, `n=2,849`; accepted subset 2,744/2,849 | Phase 4 `speech.json`; speaker-held-out TORGO dysarthria task | Dysarthria proxy, not stroke-specific. Metric reuse may need further permission review; no audio is included in this repository. |
| Contextual-risk ROC-AUC 0.8338, `n=767`, 38 positive rows | Phase 4 `metadata_context.json` | Imbalanced general-risk data, not acute triage; precision is 0.1314. |
| Seven symptom-rule checks and 16 missing-modality combinations | Phase 4 `symptoms.json` and `missing_modalities.json` | Rule/contract checks, not clinical approval or clinical validation. |
| Model artifact bundle 39,146,718 to 23,421,513 bytes; warm p50 132.2 to 40.7 ms | Stage 5 `bundle_benchmarks.json` | Historical host measurements; 10 warm samples, not a native-phone benchmark or end-to-end response time. |

Raw datasets and processed manifests under `data/` are excluded from the Git repository. Consult `docs/dataset_cards/` before interpreting these values or considering public reuse.
