# Stage 2 Error Catalog

All errors contain `code`, `message`, `details`, and `correlation_id`.

| Code | HTTP | Meaning |
| --- | --- | --- |
| `validation_error` | 422 | Request schema or domain validation failed |
| `authentication_required` | 401 | Token is missing or invalid |
| `forbidden` | 403 | Role or facility scope is insufficient |
| `not_found` | 404 | Resource is absent or inaccessible |
| `conflict` | 409 | State, version, or unique-resource conflict |
| `idempotency_conflict` | 409 | Key reused with a different canonical request |
| `idempotency_in_progress` | 409 | Non-expired identical operation is in progress |
| `precondition_required` | 428 | `If-Match` is required |
| `attachment_rejected` | 400 | Media type, size, or storage validation failed |
| `attachment_too_large` | 413 | Attachment exceeds the configured byte limit |
| `unsupported_media_type` | 415 | Attachment extension and media type are not supported |
| `assessment_unavailable` | 503 | Assessment operation could not be performed |
| `dependency_unavailable` | 503 | Required server dependency is unavailable |
| `internal_error` | 500 | Safe generic server failure |
