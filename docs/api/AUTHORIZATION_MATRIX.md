# Stage 2 Authorization Matrix

| Operation | collector | clinician | demo-admin | Object rule |
| --- | --- | --- | --- | --- |
| Current identity | yes | yes | yes | Token subject only |
| Create/update draft | yes | no | yes | Collector owns case and facility scope |
| Upload/delete attachment | yes | no | yes | Owning case and facility scope |
| Create assessment | yes | no | yes | Owning case; existing AssessmentService only on server |
| Import assessment | yes | no | no | Collector-generated envelope; owning collector and facility scope; no server inference |
| Read case/result/attachment | own facility | assigned facility | scoped demo data | Inaccessible UUIDs return not found |
| Submit case | yes | no | yes | Owning assessed case, matching ETag |
| Claim review | no | yes | no | Submitted case in clinician facility scope |
| Create/read review | no | yes | no | Claim owner; append-only review |

Plain role headers are never trusted.

The following operations require `Idempotency-Key`: case creation, assessment creation, assessment import, case submission, review claim, and final review. Draft updates and submission additionally require `If-Match`; submission requires both headers. Attachment upload accepts an optional backwards-compatible `Idempotency-Key`; the offline collector always supplies its stable outbox event UUID. Attachment deletion and draft PATCH do not currently require idempotency keys.
