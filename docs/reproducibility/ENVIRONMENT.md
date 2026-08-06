# Environment

- Python: 3.11.9
- Validated environment: `../autokeras_env` relative to the repository
- Baseline requirements: [`requirements-baseline.txt`](../../requirements-baseline.txt)
- Constraints: [`constraints-baseline.txt`](../../constraints-baseline.txt)
- Development requirements: [`requirements-dev.txt`](../../requirements-dev.txt)
- Direct project dependencies: [`pyproject.toml`](../../pyproject.toml)

The validated environment loads the canonical face Keras artifact, speech joblib pipeline, and metadata joblib pipeline. Do not modify `autokeras_env` for the release workflow; use `scripts/bootstrap_local.py` for a separate `.venv`.
