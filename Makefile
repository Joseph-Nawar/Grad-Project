PYTHON_INTERPRETER ?= python

.PHONY: install test data eda clean

install:
	$(PYTHON_INTERPRETER) -m pip install --upgrade pip
	$(PYTHON_INTERPRETER) -m pip install -r requirements.txt

test:
	$(PYTHON_INTERPRETER) -m pytest tests

data:
	$(PYTHON_INTERPRETER) -m rural_stroke_assist.dataset

eda:
	$(PYTHON_INTERPRETER) -m jupyter notebook

clean:
	$(PYTHON_INTERPRETER) -c "from pathlib import Path; import shutil; root = Path('.'); [p.unlink() for pattern in ('*.pyc', '*.pyo') for p in root.rglob(pattern)]; [shutil.rmtree(p, ignore_errors=True) for p in root.rglob('__pycache__')]; [shutil.rmtree(p, ignore_errors=True) for p in (root / '.pytest_cache', root / '.ruff_cache') if p.exists()]"
