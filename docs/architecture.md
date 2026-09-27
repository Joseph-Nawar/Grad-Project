# RuralStroke-Triage architecture

RuralStroke-Triage is a multimodal prehospital triage-support research prototype. Its architecture keeps capture, model execution, evidence fusion, persistence, and clinician review in separate, auditable boundaries.

## Assessment path

The standard collector sends case inputs to the FastAPI service. `AssessmentService` invokes independent adapters for visual/facial evidence, dysarthria-related speech evidence, contextual risk metadata, and deterministic acute symptoms. Adapters emit the shared immutable `ModalityEvidence` contract. Late fusion records branch contributions and provenance, handles missing or rejected inputs, and produces an immutable assessment result.

```text
Collector UI -> FastAPI -> AssessmentService -> modality adapters
                                      -> common ModalityEvidence contract
                                      -> late fusion and symptom safeguards
                                      -> immutable assessment snapshot
API persistence -> PostgreSQL records + managed object-store attachments
Clinician UI <- submitted snapshot and separate agree/override decision
```

The edge collector uses the same assessment and fusion boundary locally. It stores assessment events and media durably in local SQLite-backed storage while the API is unavailable, then a synchronization worker retries delivery when connectivity returns. The local outbox and central API use stable event identities and provenance to support replay and audit.

## Research-reference orchestration

The `pretrained_reference` runtime profile is the Project Template 4.1 orchestration demonstration. It joins ImageNet-pretrained MobileNetV2 visual evidence, a frozen DistilHuBERT speech representation with a logistic-regression classifier, and TabPFN v2 contextual-risk evidence through independent adapters. Deterministic acute-symptom rules remain a separate fourth branch.

The `optimized` profile is used by the Compose edge collector. It uses an optimized MobileNetV2 LiteRT artifact, MFCC plus Random Forest speech inference through ONNX Runtime, the Logistic Regression metadata artifact, and the same symptom rules. Model and runtime provenance is attached to each result. See [`config/pretrained_reference_registry.json`](../config/pretrained_reference_registry.json) and [`config/edge_runtime_registry.json`](../config/edge_runtime_registry.json).

## Review and persistence

Collector submissions create immutable assessment snapshots. The clinician application reads the submitted snapshot and records agreement or override as a separate review event; it does not rerun or edit the assessment. The Compose API uses PostgreSQL for central records and MinIO for attachments. The edge collector has separate local durable storage and a retrying synchronization worker.

See the [system architecture diagram](diagrams/system_architecture.md), [case data flow](diagrams/case_data_flow.md), [testing guide](testing/TESTING.md), and [final evaluation report](../reports/evaluation/phase4/final_complete/PHASE_4_EVALUATION_REPORT.md). These boundaries support traceability; they do not establish clinical validity.
