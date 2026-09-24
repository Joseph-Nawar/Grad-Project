# syntax=docker/dockerfile:1.7

FROM python:3.11-slim-bookworm AS api-builder
ENV VIRTUAL_ENV=/opt/venv PATH="/opt/venv/bin:$PATH" PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
RUN python -m venv "$VIRTUAL_ENV"
COPY requirements-api-runtime.txt /tmp/requirements-api-runtime.txt
RUN pip install --no-cache-dir --upgrade pip setuptools wheel \
    && pip install --no-cache-dir -r /tmp/requirements-api-runtime.txt \
    && pip uninstall -y pip setuptools wheel

FROM api-builder AS api
ENV VIRTUAL_ENV=/opt/venv PATH="/opt/venv/bin:$PATH" PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
RUN groupadd --system ruralstroke && useradd --system --gid ruralstroke --home-dir /nonexistent --shell /usr/sbin/nologin ruralstroke \
    && mkdir -p /app/runtime_data /tmp/ruralstroke \
    && chown -R ruralstroke:ruralstroke /app /tmp/ruralstroke
COPY --chown=ruralstroke:ruralstroke rural_stroke_assist /app/rural_stroke_assist
COPY --chown=ruralstroke:ruralstroke alembic /app/alembic
COPY --chown=ruralstroke:ruralstroke alembic.ini pyproject.toml /app/
COPY --chown=ruralstroke:ruralstroke config/baseline_registry.json /app/config/baseline_registry.json
COPY --chown=ruralstroke:ruralstroke models/experiments/face/trial_003_mobilenetv2_balanced_160/model.keras /app/models/experiments/face/trial_003_mobilenetv2_balanced_160/model.keras
COPY --chown=ruralstroke:ruralstroke models/experiments/speech/trial_001_mfcc_random_forest/model.pkl /app/models/experiments/speech/trial_001_mfcc_random_forest/model.pkl
COPY --chown=ruralstroke:ruralstroke models/experiments/metadata/mvp_metadata_risk_model.pkl /app/models/experiments/metadata/mvp_metadata_risk_model.pkl
COPY --chown=ruralstroke:ruralstroke docker/api-entrypoint.sh /usr/local/bin/ruralstroke-api
RUN chmod 0555 /usr/local/bin/ruralstroke-api \
    && rm -rf /usr/local/bin/pip /usr/local/bin/pip3 /usr/local/bin/pip3.11 \
        /usr/local/lib/python3.11/site-packages/pip \
        /usr/local/lib/python3.11/site-packages/pip-*.dist-info \
        /usr/local/lib/python3.11/site-packages/setuptools \
        /usr/local/lib/python3.11/site-packages/setuptools-*.dist-info \
        /usr/local/lib/python3.11/site-packages/wheel \
        /usr/local/lib/python3.11/site-packages/wheel-*.dist-info \
    && python - <<'PY'
from pathlib import Path
required = (
    Path('/app/models/experiments/face/trial_003_mobilenetv2_balanced_160/model.keras'),
    Path('/app/models/experiments/speech/trial_001_mfcc_random_forest/model.pkl'),
    Path('/app/models/experiments/metadata/mvp_metadata_risk_model.pkl'),
)
missing = [str(path) for path in required if not path.is_file()]
if missing:
    raise SystemExit(f'Missing canonical model artifacts: {missing}')
PY
USER ruralstroke
EXPOSE 8000
HEALTHCHECK --interval=10s --timeout=3s --start-period=30s --retries=12 CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health/live', timeout=2)"
ENTRYPOINT ["/usr/local/bin/ruralstroke-api"]

FROM python:3.11-slim-bookworm AS ui-builder
ENV VIRTUAL_ENV=/opt/venv PATH="/opt/venv/bin:$PATH" PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
RUN python -m venv "$VIRTUAL_ENV"
COPY requirements-ui-runtime.txt /tmp/requirements-ui-runtime.txt
RUN pip install --no-cache-dir --upgrade pip setuptools wheel \
    && pip install --no-cache-dir -r /tmp/requirements-ui-runtime.txt \
    && pip uninstall -y pip setuptools wheel

FROM ui-builder AS ui
ENV VIRTUAL_ENV=/opt/venv PATH="/opt/venv/bin:$PATH" PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
RUN groupadd --system ruralstroke && useradd --system --gid ruralstroke --home-dir /nonexistent --shell /usr/sbin/nologin ruralstroke \
    && chown -R ruralstroke:ruralstroke /app
COPY --chown=ruralstroke:ruralstroke apps /app/apps
COPY --chown=ruralstroke:ruralstroke rural_stroke_assist/__init__.py /app/rural_stroke_assist/__init__.py
COPY --chown=ruralstroke:ruralstroke rural_stroke_assist/client /app/rural_stroke_assist/client
COPY --chown=ruralstroke:ruralstroke rural_stroke_assist/capture /app/rural_stroke_assist/capture
COPY --chown=ruralstroke:ruralstroke rural_stroke_assist/inference /app/rural_stroke_assist/inference
COPY --chown=ruralstroke:ruralstroke rural_stroke_assist/quality /app/rural_stroke_assist/quality
COPY --chown=ruralstroke:ruralstroke rural_stroke_assist/ui /app/rural_stroke_assist/ui
COPY --chown=ruralstroke:ruralstroke docker/ui-entrypoint.sh /usr/local/bin/ruralstroke-ui
RUN chmod 0555 /usr/local/bin/ruralstroke-ui \
    && rm -rf /usr/local/bin/pip /usr/local/bin/pip3 /usr/local/bin/pip3.11 \
        /usr/local/lib/python3.11/site-packages/pip \
        /usr/local/lib/python3.11/site-packages/pip-*.dist-info \
        /usr/local/lib/python3.11/site-packages/setuptools \
        /usr/local/lib/python3.11/site-packages/setuptools-*.dist-info \
        /usr/local/lib/python3.11/site-packages/wheel \
        /usr/local/lib/python3.11/site-packages/wheel-*.dist-info
USER ruralstroke
EXPOSE 8501
HEALTHCHECK --interval=10s --timeout=3s --start-period=20s --retries=12 CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health', timeout=2)"
ENTRYPOINT ["/usr/local/bin/ruralstroke-ui"]

FROM api-builder AS collector-edge
ENV VIRTUAL_ENV=/opt/venv PATH="/opt/venv/bin:$PATH" PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 NUMBA_CACHE_DIR=/tmp/numba-cache
# The inherited api-builder supplies the validated Stage 3A ML runtime;
# requirements-ml.txt remains the source inventory for the local edge boundary.
COPY requirements-ml.txt /tmp/requirements-ml.txt
COPY requirements-ui-runtime.txt /tmp/requirements-ui-runtime.txt
COPY requirements-edge-runtime.txt /tmp/requirements-edge-runtime.txt
# api-builder intentionally removes the virtualenv's pip before runtime stages.
# Install the collector-only UI/edge packages directly into the inherited
# virtualenv; otherwise the shell's fallback system pip would place them in
# /usr/local, which /opt/venv/bin/python cannot import.
RUN /usr/local/bin/python -m pip install --no-cache-dir --target /opt/venv/lib/python3.11/site-packages \
        -r /tmp/requirements-ui-runtime.txt -r /tmp/requirements-edge-runtime.txt \
    && /usr/local/bin/python -m pip uninstall -y pip setuptools wheel
WORKDIR /app
RUN groupadd --system ruralstroke && useradd --system --gid ruralstroke --home-dir /nonexistent --shell /usr/sbin/nologin ruralstroke \
    && mkdir -p /app/runtime_data /var/lib/ruralstroke /tmp/ruralstroke /tmp/numba-cache \
    && chown -R ruralstroke:ruralstroke /app /var/lib/ruralstroke /tmp/ruralstroke /tmp/numba-cache
COPY --chown=ruralstroke:ruralstroke apps /app/apps
COPY --chown=ruralstroke:ruralstroke rural_stroke_assist /app/rural_stroke_assist
COPY --chown=ruralstroke:ruralstroke scripts/run_collector_sync.py /app/scripts/run_collector_sync.py
COPY --chown=ruralstroke:ruralstroke config/baseline_registry.json /app/config/baseline_registry.json
COPY --chown=ruralstroke:ruralstroke config/edge_runtime_registry.json /app/config/edge_runtime_registry.json
COPY --chown=ruralstroke:ruralstroke models/experiments/face/trial_003_mobilenetv2_balanced_160/model.keras /app/models/experiments/face/trial_003_mobilenetv2_balanced_160/model.keras
COPY --chown=ruralstroke:ruralstroke models/experiments/speech/trial_001_mfcc_random_forest/model.pkl /app/models/experiments/speech/trial_001_mfcc_random_forest/model.pkl
COPY --chown=ruralstroke:ruralstroke models/experiments/metadata/mvp_metadata_risk_model.pkl /app/models/experiments/metadata/mvp_metadata_risk_model.pkl
COPY --chown=ruralstroke:ruralstroke models/edge/face-trial-003-litert-fp32.tflite /app/models/edge/face-trial-003-litert-fp32.tflite
COPY --chown=ruralstroke:ruralstroke models/edge/speech-trial-001-onnx.onnx /app/models/edge/speech-trial-001-onnx.onnx
COPY --chown=ruralstroke:ruralstroke docker/ui-entrypoint.sh /usr/local/bin/ruralstroke-ui
RUN chmod 0555 /usr/local/bin/ruralstroke-ui \
    && find /app/rural_stroke_assist/modeling -type f ! -name 'face_inference.py' ! -name '__init__.py' -delete \
    && find /app/rural_stroke_assist/modeling -type d -name '__pycache__' -prune -exec rm -rf {} + \
    && rm -rf /usr/local/bin/pip /usr/local/bin/pip3 /usr/local/bin/pip3.11 \
        /opt/venv/lib/python3.11/site-packages/pip \
        /opt/venv/lib/python3.11/site-packages/pip-*.dist-info \
        /opt/venv/lib/python3.11/site-packages/setuptools \
        /opt/venv/lib/python3.11/site-packages/setuptools-*.dist-info \
        /opt/venv/lib/python3.11/site-packages/wheel \
        /opt/venv/lib/python3.11/site-packages/wheel-*.dist-info
USER ruralstroke
EXPOSE 8501
HEALTHCHECK --interval=10s --timeout=3s --start-period=30s --retries=12 CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health', timeout=2)"
ENTRYPOINT ["/usr/local/bin/ruralstroke-ui"]
