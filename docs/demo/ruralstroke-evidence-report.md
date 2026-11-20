# RuralStroke-Triage evidence qualification

**Baseline:** `main` at `d566550452e18d29860438c2725beed692e5de1e`

**Evidence branch:** `evidence/ruralstroke-recruiter-package`

**Purpose:** Qualify authentic engineering evidence for a recruiter-facing case study. This report does not validate a medical device or authorize clinical use.

## Executive assessment

The repository contains a substantial, inspectable research prototype: separate modality adapters, a shared evidence contract, late fusion, quality rejection and missing-input handling, a persistent offline edge workflow, an API, and an independent clinician review event. The strongest engineering evidence is the separation of the local assessment snapshot from its retryable sync outbox and from the later reviewer decision.

The most compelling public evidence should be a synthetic, non-diagnostic walkthrough of the optimized edge application, plus an architecture diagram and carefully scoped proxy/system results. No current screenshot or recording existed on this baseline. Browser capture remains pending on this branch: the documented full Compose launcher stopped before starting services when the pinned MinIO images could not be pulled, and a separate temporary image that omitted the official `collector-edge` target's inherited API runtime was discarded as an invalid reproduction environment. Existing release records explicitly mark screenshot, recording, and browser review as pending. This branch does not treat those gates as completed merely because committed tests or reports exist.

The model results are branch-level proxy experiments, not end-to-end stroke-screening performance. In particular, the visual model's 0.9818 ROC-AUC coexists with only 107/318 images accepted by the runtime quality gate. The speech model is evaluated on dysarthria data, not stroke data. Metadata is background-risk evidence, not acute triage. There is no paired patient-level multimodal evaluation, prospective study, clinical validation, or native-phone benchmark.

## Three evidence sources that must remain separate

| Source | What the repository supports | What it does not establish |
| --- | --- | --- |
| Earlier MIUCC 2026 paper | The current README says an earlier framework version was accepted for presentation; the owner identifies it as accepted and presented. | The accepted manuscript and an individual conference record are absent from this repository. Paper-specific methods, metrics, indexing, DOI, and publication status cannot be checked from the checked-in source. Do not transfer current-code metrics to the paper. |
| Current code and Phase 4 evaluation | Commit-bound evaluation reports, manifests, metrics, source registries, model cards, tests, and code are inspectable. The results below belong to this source. | The proxy metrics do not measure clinical diagnosis, triage benefit, or end-to-end system accuracy. |
| Deployment-oriented optimized edge profile | `config/edge_runtime_registry.json` selects LiteRT face, ONNX speech, the retained scikit-learn metadata model, and the same deterministic symptom rules. Stage 5 reports record bundle, parity, offline-sync, and runtime evidence. | Host benchmark timings are not native smartphone performance. Stage 5 evidence is a committed historical run; it is not a fresh execution on this branch. |

The separately named `pretrained_reference` profile is also current repository code: ImageNet MobileNetV2, frozen DistilHuBERT representation plus logistic regression, TabPFN v2, and deterministic symptoms. It is not the optimized edge profile and must not be described as the accepted paper's exact implementation without the accepted manuscript.

## Verified capabilities

| Capability | Status and inspected source | Safe interpretation |
| --- | --- | --- |
| Four evidence branches | **VERIFIED** in `rural_stroke_assist/assessment/factory.py`, the modality adapters, and `README.md`: face, dysarthria-related speech, contextual metadata, and deterministic acute symptoms. | Distinct branches with heterogeneous evidence semantics, not four calibrated stroke probabilities. |
| Image/audio quality checks and missing-input handling | **VERIFIED** in `rural_stroke_assist/inference/face_adapter.py`, `speech_adapter.py`, `rural_stroke_assist/quality/`, `assessment/service.py`, and the Phase 4 missing-modality artifacts. | Inputs can be rejected or unavailable; fusion uses remaining evidence or reports insufficient evidence. |
| Deterministic acute symptom safeguards | **VERIFIED** in `rural_stroke_assist/modules/acute_symptom_module.py` and `reports/evaluation/phase4/final_complete/symptoms.json` (seven rule-verification scenarios). | A deterministic research rule path, not clinically approved rules or clinical outcome evidence. |
| Local edge assessment and persistence | **VERIFIED** in `apps/collector_edge_app.py`, `rural_stroke_assist/offline/workflow.py`, `offline/store.py`, and `tests/e2e/test_offline_local_assessment.py`. The workflow constructs the current optimized assessment service in Compose and persists locally in SQLite. | Local assessment can proceed without a central API request. A browser capture still requires a working UI runtime. |
| Durable outbox, retries, idempotent synchronization | **VERIFIED in code and committed Stage 5 evidence**: `offline/worker.py`, `offline/store.py`, `offline/envelope.py`, `reports/production/stage5/docker_closure.json`. | Stage 5 records API-unavailable local assessment followed by `SYNCED`, preserved snapshot hash, five acknowledged outbox events, and one separate clinician review. This is historical report evidence, not a new run during this audit. |
| Separate clinician agreement/override | **VERIFIED** in `apps/clinician_app.py`, `server/application/reviews.py`, and review contracts/tests. The assessment snapshot is not mutated by the review event. | A review event can record agreement or override plus rationale; this is a workflow demonstration, not medical oversight validation. |
| Local Compose stack | Compose syntax is **VERIFIED** by `docker compose ... config --quiet`. The documented full launcher is **BLOCKED** in this environment because the pinned MinIO Docker Hub images are no longer pullable. See “Reproduction status.” | Do not claim this branch's standard full-stack launcher completed. The supported filesystem attachment backend is suitable for a separately qualified local workflow, but does not test the Compose S3/MinIO path. |

## Evaluation and exact provenance

The values below come from the final committed Phase 4 run (`run_id: 20260806T131613Z-7df48bdd`, Python 3.11.9, Windows 10, seed 42, 1,000 bootstrap iterations). They are **current repository evaluation evidence**, not the accepted paper's reported results.

| Branch / check | Result | Correct scope and limitation | Authoritative source |
| --- | --- | --- | --- |
| Visual model, direct held-out path | ROC-AUC **0.9818**, PR-AUC **0.9716**, `n=318`; sample-stratified 95% bootstrap ROC-AUC interval 0.9671–0.9918. | Public proxy-image labels; not facial droop detection, clinical accuracy, or end-to-end screening. | `reports/evaluation/phase4/final_complete/face.json`; manifest `reports/evaluation/phase4/final_complete/run_manifest.json`; `docs/model_cards/face.md`. |
| Visual runtime quality gate | **107/318 accepted (33.65%)**, 211 rejected; accepted-subset ROC-AUC 0.9607 (`n=107`). | Low coverage is material. Direct model metrics and accepted-subset metrics are different populations. | `face.json`; `face_adapter_coverage.json`; `face_rejection_by_reason.csv`. |
| Speech model, direct held-out path | ROC-AUC **0.9426**, PR-AUC **0.8933**, `n=2,849`. | Speaker-held-out TORGO dysarthria proxy; not stroke-specific, not Arabic-language performance, and not stroke probability. | `reports/evaluation/phase4/final_complete/speech.json`; `docs/model_cards/speech.md`; dataset card `docs/dataset_cards/speech.md`. |
| Speech runtime accepted subset | **2,744/2,849 accepted (96.31%)**; accepted-subset ROC-AUC **0.9512**. | The higher score is an adapter-filtered subset result, not a replacement for direct performance. | `speech.json`; Phase 4 report. |
| Contextual metadata | ROC-AUC **0.8338**, PR-AUC **0.1857**, `n=767`, only 38 positive rows; precision **0.1314**. | Imbalanced general-risk data, not acute triage; calibration and positive precision are limited. | `reports/evaluation/phase4/final_complete/metadata_context.json`; `docs/model_cards/metadata.md`; `docs/dataset_cards/metadata.md`. |
| Acute symptom rule checks | **7** deterministic scenarios; hard escalation appears in scenarios with multiple FAST signs. | Rule verification only; no learned metric and no clinical approval. | `reports/evaluation/phase4/final_complete/symptoms.json`; `docs/model_cards/acute_symptoms.md`. |
| Missing-modality contract | All **16** availability combinations exercised; four combinations also checked against real model artifacts. | Contract/robustness evidence, not multi-modal clinical evaluation. | `reports/evaluation/phase4/final_complete/missing_modalities.json`; `missing_modality_matrix.csv`; `missing_modality_real_artifact_checks.csv`. |
| Optimized edge model bundle | Selected `mixed-litert`: **23,421,513 bytes** versus baseline **39,146,718 bytes**. | Model artifact bundle only; not total app/container size. | `reports/production/stage5/stage5_summary.json`; `bundle_benchmarks.json`. |
| Optimized edge warm latency | `mixed-litert` p50 **40.66 ms**, p95 **46.91 ms**, 10 warm samples; baseline p50 132.21 ms, p95 162.27 ms. | Recorded on the Windows 10 development environment. Not end-to-end HTTP latency and not a smartphone benchmark. The optimized process used more measured RSS than baseline in the report. | `reports/production/stage5/bundle_benchmarks.json`; `conversion_environment.json`; `stage5_summary.json`. |

### Profile-to-artifact map

| Profile | Face | Speech | Metadata | Symptoms | Source |
| --- | --- | --- | --- | --- | --- |
| Original/default | MobileNetV2 Keras | MFCC + Random Forest scikit-learn | Logistic Regression scikit-learn | Same deterministic rules | `config/baseline_registry.json`, `assessment/factory.py` |
| Research reference | ImageNet MobileNetV2 | Frozen DistilHuBERT representation + logistic regression | TabPFN v2 | Same deterministic rules | `config/pretrained_reference_registry.json`, `inference/reference_runtime.py` |
| Deployment-oriented optimized | LiteRT MobileNetV2 | MFCC + Random Forest through ONNX Runtime | Retained Logistic Regression scikit-learn pipeline | Same deterministic rules | `config/edge_runtime_registry.json`, `reports/production/stage5/stage5_summary.json` |

The optimized bundle decision rejected the ONNX face alternative because it was larger and slower than LiteRT in the recorded comparison, despite lower RSS. Do not omit the trade-off if discussing the decision.

## Engineering decisions, trade-offs, and contribution evidence

| Decision / responsibility | Evidence | Recruiter-relevant explanation and caveat |
| --- | --- | --- |
| Give every modality a shared evidence contract while retaining modality-specific semantics and provenance | `rural_stroke_assist/inference/contracts.py`, `assessment/service.py`, `assessment/contracts.py`, `assessment/factory.py` | This lets the assessment record distinguish an unavailable branch, a quality rejection, and a model score instead of treating every output as interchangeable. It does not calibrate the branches into clinical probabilities. |
| Keep deterministic acute symptoms separate from learned proxy scores | `rural_stroke_assist/inference/symptom_adapter.py`, `modules/acute_symptom_module.py`, `assessment/fusion_strategy.py` | Makes the rule path inspectable and allows explicit hard-escalation handling. The rules remain unapproved and unevaluated for clinical outcomes. |
| Reject or abstain on unusable visual/audio inputs and handle missing modalities explicitly | `rural_stroke_assist/quality/`, `inference/face_adapter.py`, `inference/speech_adapter.py`, `reports/evaluation/phase4/final_complete/missing_modalities.json` | Preserves input-quality and availability state through the workflow. The visual branch's 33.65% accepted coverage is a significant usability and evaluation constraint. |
| Persist an immutable edge assessment snapshot and a separate retryable outbox; store clinician review as another event | `rural_stroke_assist/offline/workflow.py`, `offline/store.py`, `offline/worker.py`, `server/application/reviews.py`, `reports/production/stage5/docker_closure.json` | Supports replayable synchronization and an audit distinction between automated output and human review. Fresh recovery verification is still blocked by the standard stack's unavailable MinIO image. |
| Select a mixed optimized runtime rather than converting every model to one backend | `config/edge_runtime_registry.json`, `reports/production/stage5/bundle_benchmarks.json` | `mixed-litert` reduced the measured model-artifact bundle and warm p50 versus baseline in the recorded host run. It used more measured RSS, and the results are not from a native phone. |
| Attribute individual implementation work | `git shortlog -sn main` and Git commit history at baseline `d566550...` | Git currently attributes all 35 visible commits to Joseph. This is repository authorship evidence, not proof of sole conceptual contribution, an exhaustive list of personal responsibilities, or a substitute for an owner-confirmed contribution statement. Confirm which modules, experiments, product decisions, and team tasks the portfolio should attribute personally. |

Suggested verification commands after installing the documented groups:

```sh
python -m ruff check .
python -m ruff format --check .
python -m compileall -q rural_stroke_assist apps scripts tests
python -m pytest -m "not slow and not evaluation and not integration" -q
python scripts/check_release_readiness.py
python -m pytest -m "integration and not slow" -q
```

The final command is artifact-backed and may require model files and local service dependencies; report a skip only when a dependency/artifact is genuinely unavailable, and do not alter the test selection to make CI green.

## CI failure and repair

The last known GitHub Actions failure is [run 36349292415](https://github.com/Joseph-Nawar/Grad-Project/actions/runs/36349292415), at commit `d566550...`. It passed dependency installation, lint, format, and compile, then failed during fast-test collection with 16 import errors: 13 missing `fastapi`, one missing `jwt`/PyJWT, one missing Alembic `command`, and one missing `torch`.

- **Incorrect dependency group:** `.github/workflows/ci.yml` installed `requirements-baseline.txt` and `requirements-dev.txt`, but not `requirements-api-runtime.txt`. That left API, SQLAlchemy, PyJWT, and Alembic imports absent. The repository's `alembic/` namespace then shadowed any import but did not contain Alembic's PyPI `command` module. This is an incomplete install, not evidence of a migration-code defect.
- **Missing test-collection dependency:** `scripts/metadata_pretrained_foundation_experiment.py` intentionally imports `torch` before sklearn-backed modules; an unmarked contract-test module imports it during collection. The supported CPU Torch requirement existed only in the optional pretrained-reference group, which CI does not install.
- **Version incompatibility:** the baseline pins pytest 9.0.3 while the dev requirement capped pytest-asyncio below 1. The [pytest-asyncio changelog](https://pytest-asyncio.readthedocs.io/en/stable/reference/changelog.html) records pytest 9 support starting in pytest-asyncio 1.3.0.

This branch adds `requirements-ci.txt`, installs it in the fast-validation job, raises the pytest-asyncio floor to 1.3, and adds the existing synthetic `tests/e2e/test_offline_local_assessment.py` as a separate CI step. Dependency installation, Ruff checks, formatting, and compilation passed on the first branch CI run. The first full fast-test execution then exposed 17 failures after successful collection: 16 tests attempted to read ignored dataset manifests or generated compatibility reports, or a machine-local fingerprint snapshot; one additional test found a stale SHA-256 for the tracked TabPFN training context. The data/report-backed tests are now explicitly marked `evaluation`, preserving them for source-equipped runs rather than weakening assertions or committing restricted data. The TabPFN runner test also reads a prepared external checkpoint, so it is classified as an artifact-backed integration test; it is not executable in CI without that checkpoint. The TabPFN registry correction is limited to the file hash: the 3,577-row training-context fingerprint still matches `389693811457126741dccd6edbdeb892f3c17149ae69069fd7ef23b34a289f3d` under the declared column types. The tracked CSV's actual SHA-256 is `97e70099617adb9acd2797d49b7feea1b03f11270c724557a099a04c759f7672`; the former registry value was stale.

The subsequent branch Actions runs [44805941177](https://github.com/Joseph-Nawar/Grad-Project/actions/runs/44805941177) and [44805952110](https://github.com/Joseph-Nawar/Grad-Project/actions/runs/44805952110) passed dependency installation, Ruff lint/format, compilation, and release validation. The fast tier reports **288 passed, 2 skipped, 27 deselected**; the separate synthetic offline local-assessment test reports **1 passed**. The deselected cases include marked slow, evaluation, and artifact-backed integration checks; those requiring missing authorized research artifacts remain unrun. This does not establish full model-backed evaluation or the external TabPFN checkpoint test.

The complete dependency resolution was previously checked for Python 3.11 on both macOS ARM and Linux x86_64; it resolved FastAPI, Alembic, PyJWT, pytest-asyncio 1.5.0, and Linux `torch==2.8.0+cpu`. Repeating the full baseline install locally remained blocked by the large TensorFlow runtime download over the available connection, so no full local pytest result is claimed. GitHub Actions on this branch are the authoritative full CI result.

## Reproduction status

The repository documents `python scripts/stack.py up` in `docs/deployment/LOCAL_CONTAINERS.md`. I verified Compose syntax. The launcher then failed at image pulls for the exact pinned `minio/minio` and `minio/mc` images. The official Docker Hub [`minio/minio` page](https://hub.docker.com/r/minio/minio/) is now marked archived. The failure is environmental/upstream-image availability, not a test or application assertion. I did not substitute a community image or change `compose.yaml`.

An exploratory container built only from the optimized model and UI dependency lists could not import the application because it omitted TensorFlow and Librosa. This was not the repository's supported `collector-edge` image: `Dockerfile` builds that target from `api-builder`, which supplies `requirements-api-runtime.txt` including both dependencies. The exploratory image was stopped and is not used as evidence of an application defect. A full browser capture was not completed in this environment; do not publish screenshots or a video until the official edge target or equivalent declared runtime has actually been run.

After the CI repairs, I also attempted to build the official `collector-edge` target alone, which does not require MinIO. Docker successfully resolved the Python base image and began installing the pinned API/ML runtime on Linux ARM64, but package downloads were unusually slow: the 29.3 MB OpenCV wheel completed at approximately 150 KB/s after 323 seconds, and the 15.6 MB pandas wheel had just begun. I stopped the build before the additional TensorFlow, UI, LiteRT, and ONNX dependencies downloaded. No image or browser session started. This is an incomplete environment setup, not an application failure, and it means no genuine UI capture can be claimed yet. A future capture can retry the official target with a faster package mirror or prebuilt trusted image; if the API is run with the supported filesystem attachment backend to bypass MinIO, label that backend and do not present it as S3/MinIO verification.

The existing `optimized_offline_compose_e2e` entry in `reports/production/stage5/docker_closure.json` records a previous successful run where the API was unavailable during local assessment, the local snapshot survived review, and the case synchronized with hash parity after reconnection. This is explicitly **historical committed evidence**. A new screen recording or screenshot must only claim fresh verification if the current UI and workflow are actually run. If current local capture uses the supported filesystem attachment backend to avoid unavailable MinIO, label that backend change; it does not validate the S3/MinIO deployment path.

Synthetic evidence input created for this package: `docs/demo/assets/synthetic-no-face-quality-check.png` is a uniform 512×512 generated test field with no human or patient content. It is suitable only to exercise a rejected/no-face quality path. Do not present it as accepted visual input, a model example, or a clinical result. No real face image, TORGO recording, manuscript image, or dataset-derived contact sheet is used.

## Rights, privacy, and publication status

- No `LICENSE` or `NOTICE` file is present in tracked repository files. The software project has no declared license. This branch does not add one.
- All `data/raw/` paths are untracked/absent from the repository's Git tree. Dataset cards report incomplete source, license, consent, demographic, or split documentation. Do not copy dataset-derived images or audio into this evidence package.
- The [official University of Toronto TORGO page](https://www.cs.toronto.edu/~complingweb/data/TORGO/torgo.html) describes TORGO as acoustic/articulatory speech from speakers with dysarthria and states use is free for academic, non-profit purposes with cited papers required for publications. The current speech card also says source, license, and consent details are not fully documented locally. No TORGO audio is included here. Treat any use of TORGO-derived metrics in a commercial/recruiter portfolio as requiring an owner rights check first.
- No manuscript or accepted-paper PDF is tracked in this repository. The owner-confirmed status remains **accepted and presented at MIUCC 2026**. No IEEE Xplore indexing, DOI, or guaranteed publication is asserted. Manuscript metrics and figures remain unverified from this repository.
- Demo content uses synthetic demographics and a non-human image only. Keep case codes generic, omit names/contact details, and do not expose generated demo tokens or local secrets in screenshots or the PR.

## Evidence package and recommended portfolio structure

| Evidence | Candidate location | What it proves | Publication status |
| --- | --- | --- | --- |
| Collector input and quality status | `docs/demo/screenshots/` | Actual Streamlit collector fields plus rejected/missing-input feedback using synthetic inputs. | Capture only after browser check. |
| Local assessment result | `docs/demo/screenshots/` | Actual optimized-profile output, branch availability, warnings, provenance, and contribution breakdown. | Synthetic and non-diagnostic. |
| Offline queued case | `docs/demo/screenshots/` | Local workflow state while API is unavailable; should be newly captured if reproduced. | Do not claim successful recovery from this frame. |
| Reconnected case and clinician review | `docs/demo/screenshots/` | Successful sync plus independent agree/override record. | Historical report exists; do not create a new capture unless rerun. |
| Short captioned walkthrough | `docs/demo/` | The actual application workflow; only produce if the full sequence runs in the real UI. | Keep safety qualifications visible throughout. |
| Model/architecture diagram | Existing `docs/diagrams/system_architecture.mmd` | Boundaries between capture, branch adapters, fusion, local queue, API, and review. | Existing authored artifact; no manuscript figure reuse. |

If a fresh offline capture is later completed, a precise caption would be: “Research prototype using synthetic inputs. The visual, speech, metadata, and symptom branches provide separate proxy or structured evidence; outputs are not a diagnosis. This capture shows local offline persistence only. Separate historical Stage 5 evidence records later synchronization and hash parity; no fresh synchronization capture is included.” This describes engineering workflow, not clinical deployment. Show proxy metrics only after data-rights review and alongside dataset, profile, acceptance, and limitations.

### Suggested 75-second sequence

1. (0–10s) State the non-diagnostic research purpose and mark the case synthetic.
2. (10–25s) Show the real edge collector inputs and input-quality feedback.
3. (25–40s) Run the actual optimized local assessment; explain which branches are unavailable/rejected and show recorded provenance.
4. (40–52s) Show the immutable local snapshot and pending outbox while the API is offline.
5. (52–65s) Restore the API, synchronize, and show the persisted clinician agreement/override as a separate review event, only if freshly reproduced.
6. (65–75s) End with the data and clinical-validation limitations.

If any step cannot be reproduced, omit it and describe the blocker; do not fill the time with simulated UI or historical claims presented as fresh capture.

## Prioritized evidence work

1. **Complete dependency validation:** GitHub Actions must pass the unchanged fast test command plus release validation. Run the offline tests and real-artifact tests supported by the available profile. Do not run evaluation suites that require missing datasets/models or access external providers.
2. **Reproduce app evidence safely:** use synthetic-only case metadata and non-human media. Capture genuine desktop and mobile views of the collector and clinician UI. Verify API-down persistence and reconnect synchronization separately.
3. **Rights review:** get written permission/scope for dataset-derived metric reuse and verify any original project artwork before public display. Keep raw datasets, recordings, restricted manuscript material, and third-party figures out of `docs/demo/`.
4. **Portfolio handoff:** select at most four authentic screenshots and one concise captioned recording; put the demo close to the case-study summary. Link repository architecture and exact reports. Distinguish current optimized implementation evidence from the earlier paper.
5. **Future validation (not claimed here):** assemble a licensed, paired, consented clinical dataset, perform a prospective clinical study, evaluate subgroup/calibration/coverage and full-system pathways, and benchmark on target smartphones before any clinical-performance or native-device claims.
