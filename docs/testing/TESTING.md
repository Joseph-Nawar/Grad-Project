# Testing

## Tiers

- Fast unit tests: default `pytest` tests excluding `integration`, `evaluation`, and `slow` where appropriate.
- Integration tests: real canonical artifacts and cross-component contracts.
- Evaluation tests: metrics, bootstrap, robustness, reports, and release evidence.
- Release validation: documentation, catalog, claims, privacy wording, links, and version consistency.

## Commands

```powershell
python -m pytest -q
python -m pytest tests/evaluation -q
python scripts/verify_baseline.py
python -m compileall -q rural_stroke_assist apps scripts tests
python scripts/check_release_readiness.py
```

CI runs the non-slow, non-evaluation, non-integration tier because canonical model artifacts are not provisioned in CI. Real-artifact verification remains a local/manual gate.

Expected warnings include the sklearn 1.4.2/1.5.2 artifact warning and existing third-party deprecations.
