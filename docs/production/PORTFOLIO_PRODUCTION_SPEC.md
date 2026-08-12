# RuralStroke-Assist Portfolio Production Specification

Status: Stage 1 target specification. The current repository remains a research MVP; this document defines interfaces and promotion gates for later stages.

## Target roles and competencies

The target engineering audience is a junior MLE or Applied AI Engineer. The portfolio should demonstrate typed boundaries, reproducible evidence, failure-aware inference orchestration, API design, persistence integrity, security fundamentals, test discipline, and honest separation of engineering evidence from clinical claims.

## Roadmap order

The authoritative roadmap sequence is:

1. Stage 0 — Frozen research MVP
2. Stage 1 — Target engineering specification
3. Stage 2 — Production API and central persistence
4. Stage 3 — Docker Compose and deployable service stack
5. Stage 4 — Offline outbox and resilient synchronization
6. Stage 5 — Edge model conversion and parity benchmarking
7. Stage 6 — Observability, drift simulation, and alerts
8. Stage 7 — Load, security, reliability, and fault-injection evidence
9. Stage 8 — Recruiter-facing demos, case study, and external validation

| Roadmap stage | Recruiter competencies demonstrated |
| --- | --- |
| Stage 0 — Frozen research MVP | Python packaging, model/artifact contracts, modality adapters, deterministic late fusion, testable Streamlit workflow, evaluation traceability, and scientific restraint. |
| Stage 1 — Target engineering specification | Requirements engineering, architecture trade-offs, measurable targets, API contract design, threat-aware data boundaries, and evidence-backed validation tooling. |
| Stage 2 — Production API and central persistence | FastAPI boundary design, Pydantic v2 schemas, PostgreSQL transactions, idempotency, immutable records, OpenAPI discipline, and contract testing. |
| Stage 3 — Docker Compose and deployable service stack | Container health checks, real service packaging, local filesystem attachment storage, and reproducible stack startup. |
| Stage 4 — Offline outbox and resilient synchronization | SQLite drafts, synchronization intents, retry/idempotency reasoning, conflict handling, and fault-injection testing. |
| Stage 5 — Edge model conversion and parity benchmarking | Conversion boundaries, frozen parity corpora, numerical tolerance, and rollback evidence. |
| Stage 6 — Observability, drift simulation, and alerts | OpenTelemetry-compatible telemetry, drift monitoring, privacy-aware operations, alert routing, and runbooks. |
| Stage 7 — Load, security, reliability, and fault-injection evidence | Load budgets, authentication/security tests, reliability evidence, failure injection, and tested image provenance. |
| Stage 8 — Recruiter-facing demos, case study, and external validation | Recruiter-facing narrative, reproducible demos, case-study evidence, and clearly bounded external validation. |

Only Stage 1 is implemented by this change. Future stages are targets, not current features.

## Deployment profiles

1. **Local all-in-one Docker Compose demonstration.** A collector-facing service, clinician-facing application, local persistence, and the existing local inference boundary run together for a reproducible portfolio demo. Demo data is pseudonymous and non-identifiable.
2. **Offline-capable collector.** The collector runs the local `AssessmentService`, stores drafts and attachments locally, appends durable submission intents to a SQLite outbox, and uses a sync client when connectivity is available. The collector can continue an assessment while disconnected.
3. **AWS reference environment.** FastAPI containers run on ECS Fargate; RDS PostgreSQL stores transactional records; S3-compatible object storage stores attachments; Cognito/OIDC supplies identity; and a cloud clinician application reviews stored snapshots. This is a reference architecture, not a clinical deployment.

Kubernetes, real clinical deployment, native mobile applications, learned fusion, and EHR integration are out of scope. Docker, FastAPI, cloud services, authentication, synchronization, observability, and conversion are not implemented in the Stage 1 repository.

## API boundary

Business endpoints live under `/api/v1`. Operational endpoints remain unversioned: `GET /health/live`, `GET /health/ready`, and `GET /version`. Operational endpoints expose only service health, readiness, or build/version metadata; they must never expose case, identity, media, or model-input data. The API is the future system-of-record boundary; Streamlit and the offline collector are clients of the contract rather than owners of server policy.

Endpoint groups:

- `GET /health/live`, `GET /health/ready`, and `GET /version`: unversioned process, dependency, and build metadata endpoints; no case, identity, media, or model-input data.
- `GET /api/v1/identity/me` and `GET /api/v1/identity/roles`: authenticated identity and effective role claims.
- `POST /api/v1/attachments`, `GET /api/v1/attachments/{attachment_id}`, and `DELETE /api/v1/attachments/{attachment_id}`: content-addressed or opaque attachment metadata and controlled download; media bytes never appear in ordinary structured logs.
- `POST /api/v1/assessments`, `GET /api/v1/assessments/{assessment_id}`, and `GET /api/v1/assessments/{assessment_id}/result`: collector-side assessment requests and stored results. Inference is performed by the assessment service boundary, not by the clinician app.
- `POST /api/v1/cases`, `GET /api/v1/cases/{case_id}`, and `PATCH /api/v1/cases/{case_id}`: idempotent case creation and draft updates subject to state rules.
- `POST /api/v1/cases/{case_id}/submit`: creates the immutable submitted snapshot; repeated requests with the same idempotency key have one central effect.
- `GET /api/v1/cases`, `GET /api/v1/cases/{case_id}/snapshot`, and `GET /api/v1/cases/{case_id}/attachments`: clinician queue and exact submitted snapshot reads.
- `POST /api/v1/cases/{case_id}/reviews`: a clinician review records agreement or override as a separate decision; it does not edit or recompute the assessment.

The contract requires OpenAPI 3 documentation committed with the API, operation IDs, request and response schemas, authentication requirements, role requirements, examples for success and typed errors, idempotency-key semantics, pagination rules, correlation IDs, and explicit deprecation/version policy. Contract tests must run in CI before a tested image digest can be promoted.

Typed errors use a stable envelope: `code`, `message`, `details`, `correlation_id`, and optional `retryable`. Expected classes include `validation_error`, `authentication_required`, `forbidden`, `not_found`, `conflict`, `idempotency_conflict`, `attachment_rejected`, `assessment_unavailable`, `dependency_unavailable`, and `rate_limited`. Messages must not contain raw media, tokens, or identifiers beyond the minimum safe reference.

## Authentication and authorization

The roles are `collector`, `clinician`, and `demo-admin`. A collector may create and update drafts, attach evidence, run an assessment, and submit a case for the permitted facility. A clinician may list assigned submitted cases, read immutable snapshots and attachments, and create a review decision. A demo-admin may seed or remove demo data and inspect operational metadata in the demonstration environment. These permissions are server-side checks, never UI-only affordances.

The target uses OIDC/OAuth2 with JWT access tokens issued by Cognito or a compatible identity provider. The API validates issuer, audience, signature, expiry, and required role/facility claims. OIDC is preferred to custom password storage because identity lifecycle, MFA options, revocation policy, and provider audit capabilities remain outside this project’s application database. Tokens and raw claims are not written to logs.

## Offline and synchronization contract

The **Collector durable outbox** is required in Stage 4. It keeps local SQLite drafts, attachment metadata, and synchronization intents. Each item contains a client event ID, aggregate ID, operation type, payload schema version, creation time, attempt count, and next-attempt time. The sync client sends events at least once, retries retryable failures with bounded backoff, preserves items across process restarts, and moves terminal failures to a reviewable dead-letter state.

The **Server transactional outbox** is deferred unless a real downstream event consumer is later introduced. Stage 2 requires transactional, idempotent central endpoints and immutable records, but does not require a central event bus or server outbox. At-least-once delivery is the collector transport guarantee. Exactly-once effect is achieved by server-side idempotency keys and unique constraints on the client event ID, so a retry can be acknowledged without creating a second case or second submission. Duplicate payloads are safe; a reused key with a different payload returns `idempotency_conflict`. Network timeout after server acceptance is treated as an unknown outcome and retried safely.

Draft edits may merge only while the case is mutable. A submitted snapshot is immutable. Concurrent draft edits use a version or ETag and return `conflict` when the base version is stale. A new clinical or model result requires an explicit revision linked to the prior snapshot; it never rewrites the original submission. The clinician application only reads stored results and snapshots and must never rerun inference.

## Immutable submissions and revision policy

Submission stores the validated input, accepted/rejected modality status, evidence, warnings, fusion configuration, model/artifact identifiers, environment provenance, attachment references, and timestamps as one immutable snapshot. The case transitions remain explicit: draft, assessed, submitted, in review, and reviewed agreed or overridden. A review is a separate append-only decision with reviewer identity, reason, and timestamp.

Corrections before submission update the draft. Corrections after submission create a new revision with a new assessment ID and a reason; the prior snapshot remains readable. There is no silent backfill, clinician-triggered inference, or mutation of a released snapshot.

## Observability and drift

Structured logs use JSON and include timestamp, service, environment, route or operation, outcome, latency, correlation ID, request ID, actor role, case reference hash, and error code. They exclude names, addresses, national identifiers, raw JWTs, audio, images, free-text clinical notes, full request bodies, and model inputs. PII is not used as a metric label.

Metrics cover request count, error count, latency, dependency health, queue depth, outbox age and retry count, attachment rejection reason, assessment modality availability, and API authorization outcomes. Traces use OpenTelemetry-compatible context propagation across API, persistence, object storage, assessment, and sync operations. Trace attributes follow the same PII exclusions.

Stage 1 distinguishes measurable engineering drift from unavailable clinical-performance drift:

- **Input drift:** changes in input shape, missingness, file type, duration, quality scores, or feature distributions.
- **Quality drift:** changes in acceptance and rejection rates or rejection reasons by modality.
- **Prediction drift:** changes in score, band, or modality-availability distributions without asserting label truth.
- **Clinical-performance drift:** sensitivity, specificity, calibration to patient outcomes, or treatment impact. This is unavailable because the MVP has proxy datasets and no paired clinical outcome stream; it is deferred rather than estimated from proxy logs.

Alerts require a defined window, minimum sample count, baseline version, severity, owner, and suppression policy. Alerting does not authorize autonomous clinical action.

## Portfolio-demo retention and privacy assumptions

The demonstration uses only public, synthetic, or non-identifiable data. No real patient identifiers or real clinical media are permitted. The following are provisional portfolio-demo defaults, not legal or clinical policy:

```text
Cloud demo cases and attachments: 7 days
Operational logs:                 14 days
Aggregated metrics:               30 days
Local synchronized media:         deleted after a configurable grace period
Pending or failed sync records:    retained until resolved or manually deleted
```

Local demo data may be cleared between runs. These defaults are demonstration assumptions; a later implementation must document deletion, backup behavior, audit-log handling, and access review before handling any sensitive data.

The target cloud design assumes encrypted transport and storage, least-privilege service roles, private object access, and explicit deletion workflows. These are design requirements for later stages, not claims about the current MVP.

## Explicit non-goals

- Diagnosis, treatment recommendation, autonomous emergency action, or clinical decision replacement.
- Clinical sensitivity, specificity, calibration, regulatory approval, or real-world safety claims.
- Kubernetes, native mobile apps, EHR integration, learned fusion, and production clinical deployment.
- FastAPI, Docker, PostgreSQL, S3, Cognito/OIDC, synchronization, observability, or model conversion in Stage 1.
- Changing the frozen models, datasets, manifests, inference, fusion, `AssessmentService`, case persistence, Streamlit behavior, evaluation artifacts, or MVP release.
