# AWS reference deployment

The AWS reference environment is Terraform-managed and defaults to `eu-central-1` (Europe/Frankfurt). All resources consume the overridable `aws_region` variable; region-specific Availability Zones and the supported PostgreSQL 16.x RDS engine version are resolved through AWS provider data sources before planning.

## Architecture

Bootstrap owns the versioned/encrypted S3 Terraform state bucket with native S3 lockfiles, GitHub OIDC deployment identity, immutable ECR repositories, ECR scan-on-push, and lifecycle policy. Demo provisions a two-AZ VPC, public ALB subnets, private ECS subnets, private RDS subnets, one documented non-HA NAT gateway, ECS Fargate, private RDS PostgreSQL 16.x, private encrypted S3 attachments with a seven-day demo lifecycle, Cognito, Secrets Manager, CloudWatch logs, and ECS circuit-breaker rollback.

The ALB has one HTTPS listener. `/collector` and `/clinician` route to sticky Streamlit target groups; health and API routes use the API target group. RDS and S3 are not public. ECS task roles use the AWS SDK default credential chain and do not contain embedded AWS keys.

## Bootstrap and apply

```powershell
terraform -chdir=infra/terraform/bootstrap init
terraform -chdir=infra/terraform/bootstrap apply -var='github_repository=OWNER/REPOSITORY'
terraform -chdir=infra/terraform/bootstrap init -migrate-state -backend-config=backend.hcl

terraform -chdir=infra/terraform/demo init
terraform -chdir=infra/terraform/demo validate
terraform -chdir=infra/terraform/demo plan `
  -var='api_image_digest=ACCOUNT.dkr.ecr.eu-central-1.amazonaws.com/ruralstroke-assist-api@sha256:...' `
  -var='ui_image_digest=ACCOUNT.dkr.ecr.eu-central-1.amazonaws.com/ruralstroke-assist-ui@sha256:...' `
  -var='acm_certificate_arn=arn:aws:acm:eu-central-1:ACCOUNT:certificate/ID' `
  -var='demo_hostname=demo.example.org' `
  -var='cognito_domain_prefix=unique-ruralstroke-demo'
```

The migration task definition is separate from the ECS services. Deployment runs it with `alembic upgrade head` and only updates services after exit code `0`. Task definitions carry exact image digests, not mutable tags.

## Authentication and secrets

Cognito groups are `collector` and `clinician`. A minimal pre-token Lambda propagates the `custom:facility_scope` attribute to a `facilities` access-token claim. The API validates Cognito RS256/JWKS signature, issuer, client ID, optional audience, expiry, and `token_use=access`; existing facility authorization remains enforced.

OIDC client secrets, the database URL, and generated cloud credentials belong in Secrets Manager. They are injected into ECS tasks through execution-role secret access. The current Terraform resources use `random_password`, the RDS password argument, Cognito client-secret attributes, and `aws_secretsmanager_secret_version`; these sensitive values therefore enter Terraform state even though Terraform marks them sensitive and suppresses them from normal plan output. Use the encrypted, access-controlled, versioned remote state configured by Bootstrap, restrict state access, and do not commit local state. Source control, images, and logs must not contain credentials. The implementation does not claim that secret values are absent from state.

## HTTPS gate, costs, and rollback

Cloud completion requires an existing ACM certificate/domain configuration or an explicitly provisioned certificate and Route 53 record. Do not weaken the listener to HTTP. The demo is cost-conscious rather than highly available: one NAT gateway, one task per service, small RDS instance, short logs, and seven-day attachment retention. NAT, ALB, Fargate, RDS, CloudWatch, and public IPv4-related charges still apply.

ECS deployment circuit breakers roll back an unhealthy revision. Rollback uses the previous task-definition revision and exact previously tested image digest. `terraform destroy` removes the demo environment; preserve Stage 3 evidence before destroying it.

## Current gate status

Terraform source has been formatted and validated locally. A real apply, HTTPS smoke test, Cognito login, RDS/S3 workflow, and ECS E2E remain blocked until AWS credentials/tooling, an ACM/domain configuration, and a deployable ECR image bundle are supplied.
