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

CI installs `requirements-baseline.txt`, `requirements-ci.txt`, and `requirements-dev.txt`, then runs the non-slow, non-evaluation, non-integration tier plus the synthetic offline local-assessment integration test. The CI group adds the API runtime imports used during test collection and CPU PyTorch required by the standalone metadata experiment contract tests. It does not install the optional pretrained-reference model stack or run held-out model-backed evaluation. Canonical evaluation artifact verification remains a separate gate.

Expected warnings include the sklearn 1.4.2/1.5.2 artifact warning and existing third-party deprecations.
