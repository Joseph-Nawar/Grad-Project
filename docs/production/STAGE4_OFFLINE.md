# Stage 4 offline collector

The Stage 4 collector edge is local-first. The collector writes managed media to the `collector-media` named volume, keeps local workflow and synchronization state in the `collector-sqlite` named volume, and runs the existing `AssessmentService` locally. Central API availability affects only synchronization.

The normal Stage 3 services remain available for clinician review. The Stage 4 services are:

- `collector-edge`: local Streamlit collector and ML runtime on port 8503.
- `collector-sync`: one synchronization worker for the shared collector store.

`docker compose down` preserves the collector named volumes. To remove only Stage 4 collector state while preserving central PostgreSQL and MinIO data, run:

```powershell
python scripts/stack.py reset-collector
```

The existing `reset` command removes the complete local Stage 3 demonstration, including central volumes and local demo secrets.

Assessment import uses a path-free canonical envelope. Media is referenced by stable attachment UUID, logical kind/media type/size, and SHA-256. The local case UUID, optional attachment UUID, assessment UUID, and separate outbox event UUID are all stable. The event UUID is the retry idempotency key.

Connectivity and server retryable failures remain durable indefinitely with bounded exponential backoff. Permanent immutable-payload failures remain in `DEAD_LETTER`; resolution clones the work into a new local draft with new resource IDs.

## Local media retention

The synchronization worker and collector UI run cleanup with `RURALSTROKE_MEDIA_GRACE_SECONDS` (default: 86400 seconds). Cleanup requires the case to be locally `SYNCED`, the confirmed submission hash to be present, every outbox event to be `ACKED`, and every attachment to have a remote attachment ID. Pending, retrying, authentication-blocked, conflicted, and dead-letter cases are never eligible. After the grace period, only the local media bytes are removed; attachment UUIDs, checksums, media metadata, remote IDs, and synchronization metadata remain in SQLite.

Recovery controls are deliberately conservative: blocked authentication has a restore-and-retry action that re-reads the configured token while preserving event IDs; conflict and dead-letter cases can be cloned into new drafts with new resource IDs; queued cases are read-only and their immutable events are never edited.

## API synchronization contract

- `POST /api/v1/assessments/import` is collector-only and facility-scoped. It requires `Idempotency-Key`, validates the canonical envelope and approved provenance, and never calls server inference.
- `POST /api/v1/attachments` remains compatible with existing callers because `Idempotency-Key` is optional. The offline worker always supplies the stable outbox event UUID and may supply the stable attachment UUID; replay matching is based on logical metadata and checksum.
