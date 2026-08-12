# Terraform bootstrap

Bootstrap creates the versioned, encrypted, public-blocked S3 state bucket with native S3 lockfiles, the GitHub Actions OIDC deployment role, and immutable ECR repositories with scan-on-push. The first bootstrap apply is intentionally local-state bootstrapping; after the bucket exists, re-run `terraform init -migrate-state -backend-config=backend.hcl` using a private, uncommitted backend configuration.

Required input:

```powershell
terraform init
terraform apply -var='github_repository=OWNER/REPOSITORY'
terraform init -migrate-state -backend-config=backend.hcl
```

Never commit `terraform.tfstate`, `.terraform/`, backend credentials, or generated variable files.
