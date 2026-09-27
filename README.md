# RuralStroke-Triage

RuralStroke-Triage is a multimodal prehospital stroke-triage decision-support research prototype motivated by rural and resource-constrained settings. It combines visual/facial, speech-related, acute-symptom, and contextual-risk evidence through transparent late fusion, then presents an immutable assessment snapshot for clinician review. It is not a diagnostic system: branch outputs are heterogeneous evidence values, not mutually calibrated clinical stroke probabilities.

## Overview

The project explores how different data types can be brought into one auditable assessment workflow when specialist review may not be immediately available. Each branch runs through its own adapter and emits a common `ModalityEvidence` contract. The assessment service records branch status, quality results, warnings, model provenance, and fusion contributions alongside the result.

The research-reference profile makes the Project Template 4.1 orchestration objective explicit: it combines three pretrained learned data spaces (image, speech, and tabular metadata) through separate adapters and a shared late-fusion boundary. Acute symptoms remain a distinct deterministic rules branch. The engineering question is how to coordinate heterogeneous and sometimes unavailable evidence, not whether one model diagnoses stroke.

## Key capabilities

- Four evidence sources: visual/facial input, dysarthria-related speech input, acute FAST-style symptoms, and contextual risk metadata.
- Explicit missing-modality handling, renormalized weights over available evidence, and an `insufficient_evidence` state.
- Modality-specific quality checks that can reject or abstain on unsuitable image or audio inputs.
- Deterministic acute-symptom escalation, kept separate from learned branch scores.
- Research-reference and deployment-oriented model profiles, with provenance recorded per assessment.
- A local edge collector that can assess while the API is unavailable, persist cases and media locally, and synchronize its durable outbox when connectivity returns.
- FastAPI assessment and case services, Streamlit collector and clinician applications, immutable submitted snapshots, and a clinician agree/override workflow.
- Docker Compose local stack with warnings and model/evidence provenance surfaced for review.

## Architecture

The standard collector sends cases to the FastAPI service. The edge collector runs the assessment locally and stores a durable local copy; its synchronization worker retries delivery to the API when connectivity is available. The API persists central case records and attachments, and the clinician application reviews the submitted snapshot without rerunning the assessment.

Each assessment service invokes the modality adapters, which return the common evidence contract. Late fusion handles missing or rejected branches, records the weights used, and applies deterministic acute-symptom safeguards before creating the immutable result snapshot.

See the [system architecture diagram](docs/diagrams/system_architecture.md), [architecture notes](docs/architecture.md), and [case data flow](docs/diagrams/case_data_flow.md).

## Model profiles

| Profile | Visual | Speech | Contextual risk |
| --- | --- | --- | --- |
| Research-reference | ImageNet-pretrained MobileNetV2 | Frozen DistilHuBERT representation with a logistic-regression classifier | TabPFN v2 |
| Deployment-oriented | Optimized MobileNetV2 (LiteRT runtime) | MFCC features with Random Forest (ONNX runtime) | Logistic Regression |

Both profiles use deterministic acute-symptom rules. The research-reference profile demonstrates orchestration across heterogeneous pretrained learned spaces. The deployment-oriented profile was selected for the local edge runtime based on artifact and parity measurements. The profiles serve different engineering objectives; neither is clinically validated or established as the better triage model. See the [research profile registry](config/pretrained_reference_registry.json) and [optimized runtime registry](config/edge_runtime_registry.json).

## Evaluation highlights

The checked-in results are proxy-task and systems-engineering evidence. They do not measure clinical diagnostic accuracy or clinical benefit.

| Evidence | Result | Interpretation |
| --- | --- | --- |
| Visual branch, held-out proxy test set | ROC-AUC 0.9818; PR-AUC 0.9716 across 318 images | Branch-level proxy result, not end-to-end triage performance. |
| Visual runtime quality gate | 107/318 accepted (33.65%) | Low coverage is a material limitation even alongside the direct model metrics. |
| Missing-modality behavior | All 16 modality-availability combinations exercised | Contract and robustness evidence, not clinical validation. |
| Optimized edge bundle | Model artifacts reduced from 39.15 MB to 23.42 MB; warm assessment p50 from 132.2 ms to 40.7 ms | Recorded local benchmark comparison; not a native smartphone benchmark. |

The visual results come from the final held-out evaluation; bundle measurements come from the Stage 5 optimization evidence. Read the [final evaluation report](reports/evaluation/phase4/final_complete/PHASE_4_EVALUATION_REPORT.md), [optimization summary](reports/production/stage5/stage5_summary.json), and [bundle benchmarks](reports/production/stage5/bundle_benchmarks.json) for scope and detail. Fusion robustness measurements are engineering sensitivity checks, not clinical validation.

## Working application

A collector enters the four evidence types, reviews modality quality and warnings, and submits the resulting snapshot. The edge collector can complete this workflow locally during an API outage and synchronize later. A clinician then reviews the submitted inputs and immutable assessment, records agreement or an override, and adds a rationale. The review decision is stored separately from the assessment snapshot.

## Quick start

Requirements: Python `>=3.11,<3.15` and Docker with the Compose v2 plugin. The launcher uses the Python standard library, creates local demo secrets automatically, builds the service images when needed, starts the Compose stack, and waits for its health checks. No separate environment bootstrap or demo-token step is required for this route.

From the repository root:

```powershell
python scripts/stack.py up
```

The local services are available at:

| Service | URL |
| --- | --- |
| Collector | http://127.0.0.1:8501 |
| Clinician review | http://127.0.0.1:8502 |
| Edge collector | http://127.0.0.1:8503 |
| API documentation | http://127.0.0.1:8000/docs |

Check service state or stop the stack with:

```powershell
python scripts/stack.py status
python scripts/stack.py down
```

`down` stops and removes containers while preserving named data volumes. See [Getting started](docs/getting-started.md) and [Privacy and data retention](docs/privacy/PRIVACY_AND_DATA_RETENTION.md) for local data handling.

## Testing and reproducibility

The repository includes unit and contract tests, artifact-backed integration checks, held-out proxy evaluation, missing-input and quality-gate analysis, model-conversion parity checks, offline durability and synchronization tests, and Compose workflow checks. The [testing guide](docs/testing/TESTING.md) describes the available tiers. The [reproducibility notes](docs/reproducibility/REPRODUCIBILITY.md) and linked reports identify the evaluated artifacts and methods.

## Repository structure

- `apps/` - Streamlit collector, clinician, and edge collector entrypoints.
- `rural_stroke_assist/` - assessment, adapters, fusion, API, persistence, offline workflow, and UI code.
- `tests/` - unit, integration, evaluation, offline, and Compose tests.
- `config/` - model, evaluation, and repository registries.
- `models/` - reference, baseline, and optimized runtime artifacts.
- `reports/` - held-out evaluation and runtime optimization evidence.
- `docs/` - architecture, model and dataset cards, testing, reproducibility, deployment, and privacy.
- `scripts/` - local stack lifecycle and evaluation utilities.

## Research output

An earlier version of this framework was accepted for presentation at MIUCC 2026. This refers to the earlier framework version; it is not a claim of IEEE Xplore indexing.

## Limitations and responsible use

RuralStroke-Triage is a research prototype and must not be used to diagnose stroke or determine care. There is no paired patient-level multimodal clinical dataset: the visual and metadata branches use public proxy datasets, and the speech branch uses dysarthria proxy data that are not stroke-specific. Contextual metadata represents background risk rather than acute presentation.

The visual runtime quality gate accepted only 107 of 318 held-out images (33.65%). Fusion weights are manually specified and sensitivity-tested, not clinically optimized. Branch metrics describe their proxy tasks; fusion stability does not establish clinical performance. The project has no prospective rural or external clinical validation and no native smartphone performance validation.

## Project status

This repository contains the final evaluated graduation-project research prototype. The implementation is not production-ready.
