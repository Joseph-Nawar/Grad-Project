# Local containers

Stage 3 local deployment runs the API and the two Streamlit clients in separate images. PostgreSQL and MinIO are reachable only on the Compose networks; the reviewer-facing ports are API `8000`, collector `8501`, and clinician `8502`.

## Prerequisites

- Docker Desktop with Compose v2 or newer.
- Python 3.11 or newer for the standard-library-only stack command.
- The validated canonical model bundle under `models/`. A build fails closed if a canonical artifact is missing; models are never downloaded at container startup.

## One-command startup

From this directory:

```powershell
python scripts/stack.py up
```

The command creates random, ignored `.runtime-secrets/` files, builds both images, starts PostgreSQL and MinIO, waits for real health checks, initializes the attachment bucket, runs Alembic once through the `migrations` service, waits for the API, runs smoke checks, and prints:

```text
collector: http://127.0.0.1:8501
clinician: http://127.0.0.1:8502
api:       http://127.0.0.1:8000
api docs:  http://127.0.0.1:8000/docs
```

The local API uses demo JWT mode. The UI clients call the API only through `ApiClient`; they do not connect directly to PostgreSQL, MinIO, or the inference service.

## Lifecycle

```powershell
python scripts/stack.py status
python scripts/stack.py logs api
python scripts/stack.py down       # stops services and preserves named volumes
python scripts/stack.py reset      # removes demo containers, volumes, and secrets
```

`down` preserves database and attachment data. `reset` is the destructive demo reset and removes only the named Stage 3 Compose volumes plus `.runtime-secrets/`.

## Troubleshooting

- If an image build reports a missing model, restore the validated artifact bundle; do not add startup downloads.
- If migration fails, inspect `python scripts/stack.py logs migrations`, correct the database/secret issue, then run `python scripts/stack.py reset` and retry.
- If the API is healthy but a UI is unavailable, inspect the corresponding Streamlit service logs. The UI image has no TensorFlow/model-serving dependency by design.
- To inspect the dependency graph without starting containers, run `docker compose --file compose.yaml config --quiet`.

The Compose migration is deliberately one-shot. Normal API startup never runs schema migrations.
