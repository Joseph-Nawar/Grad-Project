# Testing

## Tiers

- Fast unit tests: default `pytest` tests excluding `integration`, `evaluation`, and `slow` where appropriate.
- Integration tests: real canonical artifacts and cross-component contracts. The TabPFN runner test validates the actual project context and pinned external checkpoint, so it requires the prepared checkpoint cache.
- Evaluation tests: metrics, bootstrap, robustness, reports, and release evidence. Tests that require ignored dataset manifests, generated experiment reports, or external fingerprint snapshots are marked `evaluation` and run when those authorized artifacts are provisioned.
- Release validation: documentation, catalog, claims, privacy wording, links, and version consistency.

## Commands

```powershell
python -m pytest -q
python -m pytest tests/evaluation -q
python -m pytest -m evaluation -q
python scripts/verify_baseline.py
python -m compileall -q rural_stroke_assist apps scripts tests
python scripts/check_release_readiness.py
```

CI installs `requirements-baseline.txt`, `requirements-ci.txt`, and `requirements-dev.txt`, then runs the non-slow, non-evaluation, non-integration tier plus the synthetic offline local-assessment integration test. The CI group adds the API runtime imports used during test collection and CPU PyTorch required by the standalone metadata experiment contract tests. It does not install the optional pretrained-reference model stack or run held-out model-backed evaluation. Evaluation tests that consume ignored/private `data/processed` manifests, generated `reports/experiments` artifacts, or a machine-local fingerprint snapshot remain marked `evaluation`; those source-dependent checks must be run where the approved artifacts are available. No dataset or generated report is copied into CI to make these tests pass.

Expected warnings include the sklearn 1.4.2/1.5.2 artifact warning and existing third-party deprecations.
