data "aws_availability_zones" "available" {
  state = "available"
}

data "aws_caller_identity" "current" {}

data "aws_rds_engine_version" "postgres" {
  engine                  = "postgres"
  preferred_major_targets = ["16"]
  default_only            = true
}

data "archive_file" "pre_token" {
  type        = "zip"
  source_file = "${path.module}/lambda/pre_token.py"
  output_path = "${path.module}/.terraform-pre-token.zip"
}

locals {
  availability_zones = data.aws_availability_zones.available.names
  azs                = slice(local.availability_zones, 0, 2)
  name               = "${var.project_name}-${var.environment}"
  rds_engine_version = coalesce(var.rds_engine_version_override, data.aws_rds_engine_version.postgres.version)
  private_app_subnets = {
    for index, az in local.azs : az => cidrsubnet(var.vpc_cidr, 4, index + 2)
  }
  private_db_subnets = {
    for index, az in local.azs : az => cidrsubnet(var.vpc_cidr, 4, index + 8)
  }
  public_subnets = {
    for index, az in local.azs : az => cidrsubnet(var.vpc_cidr, 4, index)
  }
}
