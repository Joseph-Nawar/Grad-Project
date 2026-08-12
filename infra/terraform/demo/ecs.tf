locals {
  api_environment = [
    { name = "RURALSTROKE_ENV", value = "aws" },
    { name = "RURALSTROKE_STORAGE_BACKEND", value = "s3" },
    { name = "RURALSTROKE_S3_BUCKET", value = aws_s3_bucket.attachments.bucket },
    { name = "RURALSTROKE_S3_REGION", value = var.aws_region },
    { name = "RURALSTROKE_AUTH_BACKEND", value = "cognito" },
    { name = "RURALSTROKE_COGNITO_ISSUER", value = "https://cognito-idp.${var.aws_region}.amazonaws.com/${aws_cognito_user_pool.main.id}" },
    { name = "RURALSTROKE_COGNITO_CLIENT_IDS", value = "${aws_cognito_user_pool_client.collector.id},${aws_cognito_user_pool_client.clinician.id}" },
    { name = "RURALSTROKE_API_VERSION", value = "stage3-aws" },
  ]
  collector_environment = [
    { name = "UI_APP", value = "collector" },
    { name = "PORT", value = "8501" },
    { name = "RURALSTROKE_API_URL", value = "http://api:8000" },
    { name = "RURALSTROKE_AUTH_MODE", value = "cognito" },
    { name = "RURALSTROKE_OIDC_ISSUER", value = "https://${var.cognito_domain_prefix}.auth.${var.aws_region}.amazoncognito.com" },
    { name = "RURALSTROKE_OIDC_CLIENT_ID", value = aws_cognito_user_pool_client.collector.id },
    { name = "RURALSTROKE_OIDC_REDIRECT_URI", value = "https://${var.demo_hostname}/collector" },
    { name = "STREAMLIT_BASE_PATH", value = "collector" },
  ]
  clinician_environment = [
    { name = "UI_APP", value = "clinician" },
    { name = "PORT", value = "8501" },
    { name = "RURALSTROKE_API_URL", value = "http://api:8000" },
    { name = "RURALSTROKE_AUTH_MODE", value = "cognito" },
    { name = "RURALSTROKE_OIDC_ISSUER", value = "https://${var.cognito_domain_prefix}.auth.${var.aws_region}.amazoncognito.com" },
    { name = "RURALSTROKE_OIDC_CLIENT_ID", value = aws_cognito_user_pool_client.clinician.id },
    { name = "RURALSTROKE_OIDC_REDIRECT_URI", value = "https://${var.demo_hostname}/clinician" },
    { name = "STREAMLIT_BASE_PATH", value = "clinician" },
  ]
}

resource "aws_ecs_task_definition" "api" {
  family                   = "${local.name}-api"
  requires_compatibilities = ["FARGATE"]
  network_mode             = "awsvpc"
  cpu                      = 2048
  memory                   = 4096
  execution_role_arn       = aws_iam_role.ecs_execution.arn
  task_role_arn            = aws_iam_role.ecs_task.arn
  container_definitions = jsonencode([{
    name         = "api"
    image        = var.api_image_digest
    essential    = true
    portMappings = [{ containerPort = 8000, protocol = "tcp" }]
    environment  = local.api_environment
    secrets      = [{ name = "RURALSTROKE_DATABASE_URL", valueFrom = aws_secretsmanager_secret.database_url.arn }]
    logConfiguration = {
      logDriver = "awslogs"
      options = {
        "awslogs-group"         = aws_cloudwatch_log_group.services["api"].name
        "awslogs-region"        = var.aws_region
        "awslogs-stream-prefix" = "ecs"
      }
    }
    healthCheck = {
      command     = ["CMD-SHELL", "python -c \"import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health/ready', timeout=2)\""]
      interval    = 30
      timeout     = 5
      retries     = 3
      startPeriod = 60
    }
  }])
}

resource "aws_ecs_task_definition" "collector" {
  family                   = "${local.name}-collector"
  requires_compatibilities = ["FARGATE"]
  network_mode             = "awsvpc"
  cpu                      = 512
  memory                   = 1024
  execution_role_arn       = aws_iam_role.ecs_execution.arn
  task_role_arn            = aws_iam_role.ecs_task.arn
  container_definitions = jsonencode([{
    name         = "collector"
    image        = var.ui_image_digest
    essential    = true
    portMappings = [{ containerPort = 8501, protocol = "tcp" }]
    environment  = local.collector_environment
    secrets      = [{ name = "RURALSTROKE_OIDC_CLIENT_SECRET", valueFrom = aws_secretsmanager_secret.oidc_collector.arn }]
    logConfiguration = {
      logDriver = "awslogs"
      options = {
        "awslogs-group"         = aws_cloudwatch_log_group.services["collector"].name
        "awslogs-region"        = var.aws_region
        "awslogs-stream-prefix" = "ecs"
      }
    }
  }])
}

resource "aws_ecs_task_definition" "clinician" {
  family                   = "${local.name}-clinician"
  requires_compatibilities = ["FARGATE"]
  network_mode             = "awsvpc"
  cpu                      = 512
  memory                   = 1024
  execution_role_arn       = aws_iam_role.ecs_execution.arn
  task_role_arn            = aws_iam_role.ecs_task.arn
  container_definitions = jsonencode([{
    name         = "clinician"
    image        = var.ui_image_digest
    essential    = true
    portMappings = [{ containerPort = 8501, protocol = "tcp" }]
    environment  = local.clinician_environment
    secrets      = [{ name = "RURALSTROKE_OIDC_CLIENT_SECRET", valueFrom = aws_secretsmanager_secret.oidc_clinician.arn }]
    logConfiguration = {
      logDriver = "awslogs"
      options = {
        "awslogs-group"         = aws_cloudwatch_log_group.services["clinician"].name
        "awslogs-region"        = var.aws_region
        "awslogs-stream-prefix" = "ecs"
      }
    }
  }])
}

resource "aws_ecs_service" "api" {
  name            = "api"
  cluster         = aws_ecs_cluster.main.id
  task_definition = aws_ecs_task_definition.api.arn
  desired_count   = 1
  launch_type     = "FARGATE"
  deployment_circuit_breaker {
    enable   = true
    rollback = true
  }
  network_configuration {
    subnets          = [for subnet in aws_subnet.private_app : subnet.id]
    security_groups  = [aws_security_group.ecs.id]
    assign_public_ip = false
  }
  load_balancer {
    target_group_arn = aws_lb_target_group.api.arn
    container_name   = "api"
    container_port   = 8000
  }
}

resource "aws_ecs_service" "collector" {
  name            = "collector"
  cluster         = aws_ecs_cluster.main.id
  task_definition = aws_ecs_task_definition.collector.arn
  desired_count   = 1
  launch_type     = "FARGATE"
  deployment_circuit_breaker {
    enable   = true
    rollback = true
  }
  network_configuration {
    subnets          = [for subnet in aws_subnet.private_app : subnet.id]
    security_groups  = [aws_security_group.ecs.id]
    assign_public_ip = false
  }
  load_balancer {
    target_group_arn = aws_lb_target_group.collector.arn
    container_name   = "collector"
    container_port   = 8501
  }
}

resource "aws_ecs_service" "clinician" {
  name            = "clinician"
  cluster         = aws_ecs_cluster.main.id
  task_definition = aws_ecs_task_definition.clinician.arn
  desired_count   = 1
  launch_type     = "FARGATE"
  deployment_circuit_breaker {
    enable   = true
    rollback = true
  }
  network_configuration {
    subnets          = [for subnet in aws_subnet.private_app : subnet.id]
    security_groups  = [aws_security_group.ecs.id]
    assign_public_ip = false
  }
  load_balancer {
    target_group_arn = aws_lb_target_group.clinician.arn
    container_name   = "clinician"
    container_port   = 8501
  }
}
