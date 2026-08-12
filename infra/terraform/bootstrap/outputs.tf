output "state_bucket_name" {
  value = aws_s3_bucket.terraform_state.bucket
}

output "github_deploy_role_arn" {
  value = aws_iam_role.github_deploy.arn
}

output "ecr_repository_urls" {
  value = { for key, repository in aws_ecr_repository.images : key => repository.repository_url }
}
