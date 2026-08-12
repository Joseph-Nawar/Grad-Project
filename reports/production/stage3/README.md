# Stage 3 production evidence

This directory contains machine-readable evidence for the Stage 3 deployment
boundary. It is intentionally separate from Stage 2 benchmark and evaluation
evidence.

`stack-evidence.json` records the locally tested image identifiers and sizes,
Compose readiness observations, container RSS samples, warm API assessment
latency, score stability, redacted environment metadata, and hashes of the
canonical baseline/model references. `evidence.schema.json` defines the stable
shape consumed by CI or later release review tooling.

The canonical generated security and build records are `trivy-api.json`,
`trivy-ui.json`, `sbom-api.json`, `sbom-ui.json`, `provenance-api.json`, and
`provenance-ui.json`. The first four names match the Stage 3 CI workflow; the
provenance records are portable captures of the corresponding BuildKit
attestations. Host paths, secret mounts, and credentials are excluded from
committed evidence.

Cloud image digests and HTTPS/Cognito/RDS/S3/ECS evidence are added only after
an actual AWS deployment has run; Terraform validation alone is not represented
as cloud completion.
