# Demonstration guide

RuralStroke-Assist is demonstrated as a local collector-to-clinician workflow
for research screening and triage support. Use only the non-identifiable demo
state and public-data media supplied by the repository.

## Prepare the demonstration

```powershell
python scripts/prepare_demo_state.py
python scripts/run_phase3_apps.py
```

Open the collector at `http://localhost:8501` and the clinician app at
`http://localhost:8502`. The local runtime directory contains SQLite case data
and managed attachments; clear it with the dry-run-first command documented in
[`docs/privacy/PRIVACY_AND_DATA_RETENTION.md`](../privacy/PRIVACY_AND_DATA_RETENTION.md).

## Suggested walkthrough

1. Review the collector inputs and run an assessment.
2. Submit the case and copy the short case reference.
3. Open the submitted case in the clinician queue.
4. Review the stored assessment and media without rerunning inference.
5. Record an agreement or a documented proposed-urgency override.

This is a demonstration of workflow integrity, not clinical validation or
autonomous decision-making. Screenshot and video capture remain pending; see
[`SHOT_LIST.md`](SHOT_LIST.md).
