variable "aws_region" {
  description = "AWS region for Stage 3 bootstrap resources."
  type        = string
  default     = "eu-central-1"
}

variable "project_name" {
  type    = string
  default = "ruralstroke-assist"
}

variable "state_bucket_name" {
  description = "Globally unique versioned S3 bucket for Terraform state."
  type        = string
  default     = "ruralstroke-assist-terraform-state-eu-central-1"
}

variable "github_repository" {
  description = "GitHub owner/repository allowed to assume the deployment role."
  type        = string
}

variable "github_branch" {
  type    = string
  default = "main"
}
