# Stage 1 Roadmap Traceability

This matrix uses the authoritative roadmap order. “Current” means present in the frozen MVP; “target” means specified but not implemented.

## Authoritative roadmap order

1. Stage 0 — Frozen research MVP
2. Stage 1 — Target engineering specification
3. Stage 2 — Production API and central persistence
4. Stage 3 — Docker Compose and deployable service stack
5. Stage 4 — Offline outbox and resilient synchronization
6. Stage 5 — Edge model conversion and parity benchmarking
7. Stage 6 — Observability, drift simulation, and alerts
8. Stage 7 — Load, security, reliability, and fault-injection evidence
9. Stage 8 — Recruiter-facing demos, case study, and external validation

| Roadmap stage | Competency and requirement | Current evidence | Promotion gate |
| --- | --- | --- | --- |
| Stage 0 — Frozen research MVP | Typed four-input local assessment, deterministic fusion, immutable local workflow, reproducible proxy evaluation | `rural_stroke_assist/assessment`, `rural_stroke_assist/cases`, Phase 4 evidence, existing tests | Preserve frozen artifacts and baseline verification |
| Stage 1 — Target engineering specification | Requirements, architecture trade-offs, measurable targets, API and security boundaries, validation tooling | This specification, target YAML, baseline builder, validator, focused tests | Stage 1 validator and full repository verification |
| Stage 2 — Production API and central persistence | FastAPI, feature-oriented boundaries, Pydantic v2 API schemas, SQLAlchemy 2.x, Alembic, Psycopg 3, real PostgreSQL integration, typed errors, idempotent transactions, immutable records | Server implementation, OpenAPI snapshot, real PostgreSQL migration/workflow checks, concurrency check, API benchmark, and app architecture tests | Preserve the tested API contract, rerun PostgreSQL transaction/concurrency checks, and retain evidence for each release candidate |
| Stage 3 — Docker Compose and deployable service stack | Local Compose demonstration, health/version endpoints, S3-compatible attachment backend, separated tested images | Stage 3A local deployment: complete and verified. Stage 3B AWS reference deployment: implementation complete and static validation complete; real AWS deployment/cloud E2E deferred because of an external AWS account/payment prerequisite. Overall Stage 3 remains open until the Stage 3B cloud gate passes. | Preserve the clean local gate and complete the HTTPS AWS deployment/E2E gate |
| Stage 4 — Offline outbox and resilient synchronization | Collector durable outbox, retry/restart safety, conflict handling, at-least-once transport, exactly-once central effect | `rural_stroke_assist/offline`, collector-edge/collector-sync Compose targets, API import/idempotency contracts, and `reports/production/stage4/` evidence | Complete: real Stage 3A disconnect → local assessment → reconnect → exactly-once sync → clinician review gate passed; 50-cycle loss campaign and 100-replay campaign passed |
| Stage 5 — Edge model conversion and parity benchmarking | Explicit conversion boundary, frozen parity corpus, numerical tolerance, rollback evidence | Not implemented; current inference artifacts remain frozen | Conversion parity report and approved rollback gate |
| Stage 6 — Observability, drift simulation, and alerts | Structured logs, metrics, traces, input/quality/prediction drift simulation, alert routing, PII exclusions | Not implemented; Phase 4 artifacts are evaluation evidence only | Synthetic alert delay test and telemetry contract review |
| Stage 7 — Load, security, reliability, and fault-injection evidence | Candidate latency/memory budgets, authentication tests, reliability evidence, fault injection, tested image digest | Not implemented | Load report, security test report, reliability report, and deployable digest gate |
| Stage 8 — Recruiter-facing demos, case study, and external validation | Portfolio narrative, reproducible demo, case study, and bounded external validation | Not implemented | Demo checklist, case-study evidence review, and explicit claim boundary review |

## Stage 3 execution status

### Stage 3A — Local deployment

**COMPLETE AND VERIFIED**

### Stage 3B — AWS reference deployment

**IMPLEMENTATION COMPLETE**  
**STATIC VALIDATION COMPLETE**  
**REAL AWS DEPLOYMENT / CLOUD E2E DEFERRED**

Reason: **External AWS account/payment prerequisite.**

This is an execution sequencing decision, not a renumbering or architectural change. Stages 4–8 may proceed against the verified Stage 3A local Compose/API/PostgreSQL/MinIO environment; they are not blocked by the deferred external cloud gate.

Real Stage 3B AWS deployment remains a mandatory completion gate before the final project/release is considered fully complete.

## Current execution order

Completed:

- Stage 0
- Stage 1
- Stage 2
- Stage 3A
- Stage 3B implementation/static validation

Current:

- Stage 5 → Stage 6 → Stage 7 → Stage 8 preparation

Stage 4 execution status:

**COMPLETE AND VERIFIED**

Evidence: `reports/production/stage4/compose-e2e.json`, `reports/production/stage4/sync_campaigns.json`, and `reports/production/stage4/policy-tests.json`.

Deferred external gate:

- Return to Stage 3B for real AWS deployment and cloud verification.

Finalization:

- Run any AWS-specific Stage 7 checks and complete final Stage 8 deployment evidence.

## Boundary clarifications

- The **Collector durable outbox** is a Stage 4 local SQLite mechanism for offline synchronization intents and safe retries.
- The **Server transactional outbox** is deferred unless a real downstream event consumer is introduced. Stage 2 requires transactional, idempotent central endpoints and immutable records, not a central event bus or server outbox.
- The clinician application reads stored immutable snapshots and records separate append-only reviews; it never reruns inference.
- Missing modality remains unavailable evidence, never score zero. Known modality failures remain isolated from successful modalities.

## Stage 1 acceptance criteria

- The baseline JSON is reproducibly generated from existing Phase 4 JSON/CSV evidence without running evaluation.
- Every measured baseline in `config/production_targets.yaml` points to evidence and has an environment, unit, method, rationale, and roadmap stage.
- Candidate budgets are visibly unverified and carry a promotion gate tied to the correct future stage.
- The validator rejects missing sections, invalid statuses, incomplete numeric targets, missing measured evidence, unsafe claims, absolute paths, future-as-current wording, contradictory retention statements, roadmap regressions, outbox conflation, endpoint versioning errors, and unresolved Stage 2 decisions.
- No current MVP model, dataset, manifest, runtime path, case store, app, evaluation result, or release manifest is edited.

## Stage 2 open decisions

No Stage 2 technology decision remains open in Stage 1. Deployment-specific Cognito provisioning, image signing, cloud signed URLs, and infrastructure configuration remain intentionally deferred to their appropriate later stages.
