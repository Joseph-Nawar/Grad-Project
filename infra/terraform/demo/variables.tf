variable "aws_region" {
  description = "AWS region for the Stage 3 demo environment."
  type        = string
  default     = "eu-central-1"
}

variable "project_name" {
  type    = string
  default = "ruralstroke-assist"
}

variable "environment" {
  type    = string
  default = "demo"
}

variable "api_image_digest" {
  description = "Exact ECR API image reference including @sha256 digest captured by CI."
  type        = string
  validation {
    condition     = can(regex("@sha256:[0-9a-f]{64}$", var.api_image_digest))
    error_message = "api_image_digest must be an exact image reference ending in @sha256:<64 hex characters>."
  }
}

variable "ui_image_digest" {
  description = "Exact ECR UI image reference including @sha256 digest captured by CI."
  type        = string
  validation {
    condition     = can(regex("@sha256:[0-9a-f]{64}$", var.ui_image_digest))
    error_message = "ui_image_digest must be an exact image reference ending in @sha256:<64 hex characters>."
  }
}

variable "acm_certificate_arn" {
  description = "Existing ACM certificate ARN covering the HTTPS demo hostname."
  type        = string
}

variable "demo_hostname" {
  description = "HTTPS hostname used for Cognito callback URLs and smoke checks."
  type        = string
}

variable "vpc_cidr" {
  type    = string
  default = "10.42.0.0/16"
}

variable "rds_instance_class" {
  type    = string
  default = "db.t4g.micro"
}

variable "rds_engine_version_override" {
  description = "Optional exact PostgreSQL 16.x version recorded by the eu-central-1 preflight."
  type        = string
  default     = null
}

variable "cognito_domain_prefix" {
  type = string
}

variable "github_deploy_role_arn" {
  description = "Bootstrap-created GitHub OIDC deployment role ARN."
  type        = string
}
