# RuralStroke synthetic evidence shot list

All captures must show real screens from the current application and use only the synthetic non-human test field and synthetic case values. Do not include dataset images, TORGO recordings, manuscript figures, or patient identifiers.

| Order | Capture | Required proof | Status |
| --- | --- | --- | --- |
| 1 | Collector input and quality status | Synthetic case inputs and actual quality/missing-evidence feedback. | PENDING |
| 2 | Local assessment | Current optimized profile, branch availability, warnings, and recorded model provenance. | PENDING |
| 3 | Offline queue | Case and assessment remain local while the API is unavailable. | PENDING |
| 4 | Reconnected and clinician review | API sync completes, then clinician agreement/override persists as a separate event. | PENDING |
| 5 | Mobile collector view | Same app at a narrow viewport; no horizontal clipping of primary fields or status. | PENDING |

The corresponding evidence report is `docs/demo/ruralstroke-evidence-report.md`. Remove a shot from this list rather than simulating a successful result when a step cannot be reproduced. Keep at least one explicit **PENDING** item until the separately required rights and clinical-safety reviews are resolved.
