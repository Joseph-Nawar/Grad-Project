resource "aws_cognito_user_pool" "main" {
  name                     = local.name
  username_attributes      = ["email"]
  auto_verified_attributes = ["email"]
  account_recovery_setting {
    recovery_mechanism {
      name     = "verified_email"
      priority = 1
    }
  }
  schema {
    name                = "facility_scope"
    attribute_data_type = "String"
    mutable             = true
    required            = false
  }
  lambda_config {
    pre_token_generation = aws_lambda_function.pre_token.arn
  }
}

resource "aws_cognito_user_pool_domain" "main" {
  domain       = var.cognito_domain_prefix
  user_pool_id = aws_cognito_user_pool.main.id
}

resource "aws_cognito_user_group" "collector" {
  name         = "collector"
  user_pool_id = aws_cognito_user_pool.main.id
  precedence   = 20
}

resource "aws_cognito_user_group" "clinician" {
  name         = "clinician"
  user_pool_id = aws_cognito_user_pool.main.id
  precedence   = 10
}

resource "aws_cognito_user_pool_client" "collector" {
  name                                 = "${local.name}-collector"
  user_pool_id                         = aws_cognito_user_pool.main.id
  generate_secret                      = true
  allowed_oauth_flows_user_pool_client = true
  allowed_oauth_flows                  = ["code"]
  allowed_oauth_scopes                 = ["openid", "email", "profile"]
  callback_urls                        = ["https://${var.demo_hostname}/collector"]
  logout_urls                          = ["https://${var.demo_hostname}/collector"]
  supported_identity_providers         = ["COGNITO"]
}

resource "aws_cognito_user_pool_client" "clinician" {
  name                                 = "${local.name}-clinician"
  user_pool_id                         = aws_cognito_user_pool.main.id
  generate_secret                      = true
  allowed_oauth_flows_user_pool_client = true
  allowed_oauth_flows                  = ["code"]
  allowed_oauth_scopes                 = ["openid", "email", "profile"]
  callback_urls                        = ["https://${var.demo_hostname}/clinician"]
  logout_urls                          = ["https://${var.demo_hostname}/clinician"]
  supported_identity_providers         = ["COGNITO"]
}

resource "aws_secretsmanager_secret_version" "oidc_collector" {
  secret_id     = aws_secretsmanager_secret.oidc_collector.id
  secret_string = aws_cognito_user_pool_client.collector.client_secret
}

resource "aws_secretsmanager_secret_version" "oidc_clinician" {
  secret_id     = aws_secretsmanager_secret.oidc_clinician.id
  secret_string = aws_cognito_user_pool_client.clinician.client_secret
}

data "aws_iam_policy_document" "lambda_trust" {
  statement {
    effect  = "Allow"
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["lambda.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "pre_token" {
  name               = "${local.name}-cognito-pre-token"
  assume_role_policy = data.aws_iam_policy_document.lambda_trust.json
}

resource "aws_iam_role_policy_attachment" "pre_token_logs" {
  role       = aws_iam_role.pre_token.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
}

resource "aws_lambda_function" "pre_token" {
  function_name    = "${local.name}-cognito-pre-token"
  role             = aws_iam_role.pre_token.arn
  handler          = "pre_token.lambda_handler"
  runtime          = "python3.12"
  filename         = data.archive_file.pre_token.output_path
  source_code_hash = data.archive_file.pre_token.output_base64sha256
  timeout          = 5
}

resource "aws_lambda_permission" "pre_token" {
  statement_id  = "AllowCognitoPreToken"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.pre_token.function_name
  principal     = "cognito-idp.amazonaws.com"
  source_arn    = aws_cognito_user_pool.main.arn
}
