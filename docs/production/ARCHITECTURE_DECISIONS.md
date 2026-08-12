# Stage 1 Architecture Decisions

These decisions describe the target engineering direction only. They do not imply that the target services exist in the frozen MVP.

## API-first central services

**Decision:** Put future central workflow, identity, attachment, assessment, submission, and review behavior behind a versioned `/api/v1` contract.

**Rationale:** A stable boundary lets the local collector, cloud clinician application, demo profile, and future automation share typed behavior. It also makes authorization, idempotency, and contract testing server responsibilities.

**Rejected alternative:** Keep Streamlit as the system of record. That couples policy to UI sessions, prevents independent clients, and makes synchronization and server-side authorization difficult to verify.

## Local offline inference

**Decision:** The offline-capable collector runs the existing local `AssessmentService` and stores drafts and outbox records locally.

**Rationale:** Rural connectivity can be intermittent; collection must remain useful without a network round trip. Existing local model loading and explicit missing-modality behavior are the current evidence boundary.

**Rejected alternative:** Require online inference for every assessment. That makes the workflow unavailable during disconnection and turns transport availability into an assessment prerequisite.

## AWS as the reference cloud

**Decision:** Use AWS for the reference environment: ECS Fargate, RDS PostgreSQL, S3-compatible object storage, and Cognito/OIDC.

**Rationale:** The combination demonstrates managed compute, relational persistence, durable object storage, and a standard identity provider with a clear portfolio story. It is a reference target, not a clinical deployment recommendation.

**Rejected alternative:** Treat a generic cloud or self-hosted VM as the only target. That leaves identity, managed persistence, and operational boundaries underspecified for the portfolio.

## ECS Fargate instead of Kubernetes

**Decision:** Use ECS Fargate for the reference containers.

**Rationale:** The target is a small bounded service set, so managed container tasks provide a smaller operational surface and a clearer promotion path than managing a Kubernetes control plane.

**Rejected alternative:** Kubernetes. It adds cluster, ingress, scheduling, and upgrade complexity that is not justified by the Stage 1 portfolio scale or requirements.

## PostgreSQL and S3-compatible storage

**Decision:** Use PostgreSQL for transactional case, identity-reference, idempotency, and review records; use S3-compatible storage for attachments. A server transactional outbox is deferred unless a real downstream event consumer is later introduced.

**Rationale:** Relational constraints and transactions support immutable snapshots and unique idempotency effects. Object storage handles media separately, with metadata and access policy in the database.

**Rejected alternative:** Store media blobs in PostgreSQL. This entangles large object transfer with transactional tables and complicates lifecycle and download controls. SQLite remains the local collector store, not the central multi-client store.

## OIDC instead of custom password storage

**Decision:** Use OIDC/OAuth2 JWTs from Cognito or a compatible provider.

**Rationale:** The application should validate delegated identity and role claims rather than own password hashing, reset, MFA, and account recovery flows.

**Rejected alternative:** Implement custom passwords in the application database. It increases security-sensitive code and operational responsibility without adding portfolio value.

## Durable outbox and idempotency

**Decision:** The Collector durable outbox persists submission intents locally before synchronization, delivers them at least once, and makes server effects idempotent by client event ID and idempotency key. The Server transactional outbox is deferred unless a real downstream event consumer is later introduced.

**Rationale:** Crashes and timeouts make exactly-once transport unrealistic. Durable outbox plus unique server keys gives retry safety and exactly-once effect for case creation and submission.

**Rejected alternative:** Best-effort fire-and-forget HTTP. It loses work on process or network failure and cannot distinguish an accepted request from an unreceived request after a timeout.

## Stage 2 selected implementation direction

Stage 2 decisions are resolved as follows; they are implementation targets, not current features:

| Area | Selected direction |
| --- | --- |
| API framework | FastAPI |
| API organization | Feature-oriented API/application/infrastructure boundaries |
| API schemas | Pydantic v2 schemas separate from domain contracts |
| ORM | SQLAlchemy 2.x |
| Migrations | Alembic |
| PostgreSQL driver | Psycopg 3 |
| Integration database | Real PostgreSQL container, not SQLite emulation |
| API tests | `httpx.AsyncClient` with ASGI transport |
| Pagination | Cursor pagination using stable `(created_at, id)` ordering |
| Idempotency | Dedicated idempotency table scoped by actor/operation, storing key, request hash, response snapshot, timestamps, and expiry |
| Attachment identity | Opaque UUID plus SHA-256 checksum |
| Local attachment backend | Filesystem implementation behind a storage protocol |
| Cloud attachment backend | S3 implementation deferred to deployment stage |
| Authentication tests | Dependency-injected verified claims or test JWT verifier |
| Production authentication | Cognito-compatible OIDC/JWKS verifier in the cloud stage |
| OpenAPI validation | Generated schema snapshot plus operation-ID and contract tests |

Deployment-specific Cognito provisioning, image signing, cloud signed URLs, and infrastructure configuration remain deferred to their appropriate later stages. Stage 2 requires transactional, idempotent central endpoints and immutable records; it does not require a central event bus or server transactional outbox.

## OpenTelemetry-compatible observability

**Decision:** Instrument future services with structured logs, metrics, and OpenTelemetry-compatible traces, with explicit PII exclusions.

**Rationale:** Correlation across API, sync, persistence, object storage, and assessment is needed to diagnose reliability and drift. OpenTelemetry-compatible data keeps the backend replaceable.

**Rejected alternative:** Ad hoc print statements and vendor-only telemetry. They are hard to correlate, difficult to test, and create avoidable lock-in.

## Demo-only non-identifiable data

**Decision:** Portfolio demonstrations use pseudonymous, non-identifiable, public-data or synthetic cases and disposable retention.

**Rationale:** The MVP is not clinically validated or approved for real patient data. Demo data is enough to demonstrate engineering behavior without implying a clinical deployment.

**Rejected alternative:** Use real patient data to make the demo look realistic. That creates privacy, consent, governance, and security obligations outside this roadmap stage.

## Clinician read-only inference boundary

**Decision:** The clinician application reads the immutable submitted snapshot and records a separate review; it never reruns inference.

**Rationale:** Review must be reproducible and must refer to what the collector submitted. A new result requires an explicit versioned reassessment workflow.

**Rejected alternative:** Recompute on every clinician page load. Model versions, artifacts, and runtime state can change, causing the displayed result to differ silently from the submitted record.


