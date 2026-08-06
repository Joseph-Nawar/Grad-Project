# Privacy and data retention

RuralStroke-Assist is a local demonstration. SQLite case records and managed media are stored under the configured `runtime_data/` directory. Attachments are kept outside SQLite and referenced by generated relative paths.

Use only pseudonymous case codes and public, non-identifying demonstration media. Do not enter full names, national identifiers, addresses, or other unnecessary personal data. The project does not provide production authentication, encryption at rest, cloud synchronization, audit compliance, or clinical data governance.

Retention is manual. Review and clear demonstration data after use with:

```powershell
python scripts/clear_runtime_data.py
```

The command defaults to a dry run. Actual deletion requires `--confirm` and is restricted to the repository's `runtime_data/` directory. Never use it with a system or user-data path.
