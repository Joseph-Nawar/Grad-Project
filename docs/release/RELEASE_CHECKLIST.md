# Release checklist

## Automated gates

- [ ] `python -m pytest -q`
- [ ] `python scripts/verify_baseline.py`
- [ ] `python -m compileall -q rural_stroke_assist apps scripts tests`
- [ ] `python scripts/check_release_readiness.py`
- [ ] `python -m ruff check .`
- [ ] `python -m ruff format --check .`

## Manual gates

- [ ] Review the collector-to-clinician workflow with seeded demonstration data.
- [ ] Confirm the local runtime directory contains no unintended data.
- [ ] Complete clinical wording review; current status is PENDING.
- [ ] Review screenshots and video capture listed in [`docs/demo/SHOT_LIST.md`](../demo/SHOT_LIST.md).
- [ ] Manually inspect the final Phase 4 claim index and report.
- [ ] Create any Git tag or release manually only after all gates pass.

This release is a research MVP, not clinically validated.
