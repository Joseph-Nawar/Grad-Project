# Privacy and data retention

RuralStroke-Triage is a local research demonstration. The standard collector and clinician applications exchange data with the FastAPI service. In the Compose stack, PostgreSQL holds central case records and MinIO holds managed attachments. The edge collector stores cases, assessment snapshots, and media in local SQLite-backed storage and synchronizes its durable outbox to the API when connectivity returns. The default Compose setup is local; it does not upload data to an externally hosted service.

Use pseudonymous case codes and public, non-identifying demonstration media. Do not enter full names, national identifiers, addresses, or other unnecessary personal data. This prototype does not provide production identity management, encryption-at-rest guarantees, clinical data governance, or compliance controls.

Compose data is stored in named Docker volumes and survives `python scripts/stack.py down`. The command `python scripts/stack.py reset` removes the Compose volumes and generated local demo secrets; this permanently erases the local demonstration data. Use it only when that is intended.

The separate `python scripts/clear_runtime_data.py` utility applies to repository-local `runtime_data/` files and does not clear Compose named volumes. It defaults to a dry run, requires `--confirm` for deletion, and is restricted to the repository `runtime_data/` directory.
