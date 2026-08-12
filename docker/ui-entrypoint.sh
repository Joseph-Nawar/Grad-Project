#!/bin/sh
set -eu

app="${UI_APP:-collector}"
case "$app" in
  collector) script="apps/collector_app.py" ;;
  collector-edge) script="apps/collector_edge_app.py" ;;
  clinician) script="apps/clinician_app.py" ;;
  *) echo "UI_APP must be collector, collector-edge, or clinician" >&2; exit 2 ;;
esac

base_path="${STREAMLIT_BASE_PATH:-}"
set -- python -m streamlit run "$script" \
  --server.address "${HOST:-0.0.0.0}" \
  --server.port "${PORT:-8501}" \
  --server.headless true \
  --server.enableCORS false \
  --server.enableXsrfProtection true
if [ -n "$base_path" ]; then
  set -- "$@" --server.baseUrlPath "$base_path"
fi
exec "$@"
