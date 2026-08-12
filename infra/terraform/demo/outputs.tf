output "aws_region" {
  value = var.aws_region
}

output "alb_dns_name" {
  value = aws_lb.main.dns_name
}

output "https_base_url" {
  value = "https://${var.demo_hostname}"
}

output "ecs_cluster_name" {
  value = aws_ecs_cluster.main.name
}

output "migration_task_definition_arn" {
  value = aws_ecs_task_definition.migrations.arn
}

output "attachment_bucket_name" {
  value = aws_s3_bucket.attachments.bucket
}

output "cognito_user_pool_id" {
  value = aws_cognito_user_pool.main.id
}
