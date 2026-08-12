# CI/CD

## Pull requests and pushes

`.github/workflows/ci-stage3.yml` keeps existing validation and adds:

- Python tests, baseline checks, and OpenAPI generation validation.
- Terraform format/init/validate.
- Separate API/UI Docker builds with BuildKit SBOM and provenance flags.
- Non-root and image-boundary inspection.
- Trivy `0.58.2` image scanning. Remediable CRITICAL findings fail CI; HIGH/MEDIUM findings are recorded for later security work.
- Docker SBOM export.
- Clean Compose startup and collector-to-clinician E2E when the canonical model artifact bundle is available in the build context.

Actions used for checkout and Python setup are pinned to full commit SHAs. Trivy is invoked by its pinned release image rather than a floating action tag.

## Controlled demo deployment

`.github/workflows/publish-demo-images.yml` is the controlled build-once/push workflow: it runs only for a release tag or manual dispatch in the protected `stage3-demo` environment, builds the API/UI targets once with SBOM/provenance, tests and scans those exact local image references, pushes those same bytes, and records the ECR digests. `.github/workflows/deploy-demo.yml` then runs only through `workflow_dispatch` in the protected `stage3-demo` environment. It uses GitHub OIDC to assume the bootstrap-created AWS role; long-lived AWS keys are not supported.

The release sequence is:

```text
green CI
→ build and scan exact images
→ push the same image bytes to ECR
→ capture ECR digests
→ run the ECS migration task
→ require migration exit code 0
→ update API/collector/clinician task-definition revisions by exact digest
→ wait for ECS stability/circuit-breaker protection
→ run HTTPS smoke checks
```

The workflow validates `@sha256:<64 hex characters>` image references and verifies those digests in ECR before migration or service updates.

## Release and rollback

Keep the CI image digest, SBOM, provenance, Trivy output, migration task result, and ECS task-definition ARNs together in the Stage 3 release record. Roll back by selecting the last known-good exact task-definition revisions; do not retag a mutable image. If migration fails, deployment stops before service updates.

## Troubleshooting

- OIDC failure: verify the GitHub repository/branch or protected environment subject matches the bootstrap trust policy.
- ECR mismatch: compare the CI digest with `aws ecr describe-images` and the ECS task definition; do not proceed if they differ.
- Migration failure: inspect the one-off ECS task logs and correct schema/configuration before retrying.
- HTTPS smoke failure: inspect ALB target health, Streamlit base paths, ACM hostname coverage, and Cognito callback URLs.
