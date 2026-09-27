# Getting started

The supported local demonstration path starts the API, collector, clinician review, and offline-capable edge collector with Docker Compose.

## Requirements

- Python `>=3.11,<3.15`.
- Docker Engine or Docker Desktop with the Compose v2 plugin.

The stack launcher uses the Python standard library. It generates local demo secrets and builds the service images when required; no separate Python dependency installation, environment file, or demo-token setup is needed for this route.

## Start the stack

Run from the repository root:

```powershell
python scripts/stack.py up
```

The launcher waits for service health checks. Open:

- Collector: <http://127.0.0.1:8501>
- Clinician review: <http://127.0.0.1:8502>
- Edge collector: <http://127.0.0.1:8503>
- API documentation: <http://127.0.0.1:8000/docs>

Use the following commands to inspect or stop the stack:

```powershell
python scripts/stack.py status
python scripts/stack.py down
```

`down` preserves the named PostgreSQL, attachment-store, and edge-collector volumes. `python scripts/stack.py reset` removes the Compose volumes and generated local secrets; use it only when you intend to erase the local demonstration data.

## Workflow

The standard collector sends the assessment request to the API. The edge collector assesses locally, stores the case and media in a durable SQLite-backed outbox, and synchronizes after the API becomes reachable. Clinicians review centrally submitted snapshots and record agreement or override separately.

For storage and retention details, see [Privacy and data retention](privacy/PRIVACY_AND_DATA_RETENTION.md). For service boundaries, see [Architecture](architecture.md) and the [system diagram](diagrams/system_architecture.md).
