# Terraform demo environment

This reference environment uses the default AWS region `eu-central-1`, but every AWS resource receives `var.aws_region`; override that variable without changing resource code. The `aws_rds_engine_version.postgres` data source selects the default supported PostgreSQL 16.x engine version in the selected region. Before an apply, run the documented preflight and set `rds_engine_version_override` to the exact tested minor version if a portfolio release needs a stable pin.

The environment contains two-AZ public ALB subnets, two-AZ private ECS subnets, two-AZ private RDS subnets, one cost-conscious NAT gateway, an ECS Fargate cluster, API/collector/clinician services, a separate migration task definition, HTTPS ALB routing, sticky Streamlit target groups, private encrypted S3 attachments with a seven-day demo lifecycle, private RDS PostgreSQL 16.x, Cognito groups and OIDC clients, a narrow pre-token facility-scope trigger, Secrets Manager, CloudWatch logs, task/execution roles, and ECS circuit-breaker rollback.

Apply requires:

```powershell
terraform init
terraform validate
terraform plan -var='api_image_digest=ACCOUNT.dkr.ecr.eu-central-1.amazonaws.com/ruralstroke-assist-api@sha256:...' -var='ui_image_digest=ACCOUNT.dkr.ecr.eu-central-1.amazonaws.com/ruralstroke-assist-ui@sha256:...' -var='acm_certificate_arn=arn:aws:acm:...' -var='demo_hostname=demo.example.org' -var='cognito_domain_prefix=unique-ruralstroke-demo'
```

The HTTPS certificate/domain inputs are deliberate gates. Do not replace them with HTTP. RDS and S3 are not public. Destroying the demo removes its RDS/S3/Cognito/ECS/ALB resources; export evidence before destroy.
