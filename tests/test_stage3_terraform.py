from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).parents[1]
BOOTSTRAP = ROOT / "infra" / "terraform" / "bootstrap"
DEMO = ROOT / "infra" / "terraform" / "demo"


def _text(directory: Path) -> str:
    return "\n".join(path.read_text(encoding="utf-8") for path in sorted(directory.glob("*.tf")))


def test_terraform_has_bootstrap_and_demo_roots() -> None:
    assert BOOTSTRAP.is_dir()
    assert DEMO.is_dir()
    assert (BOOTSTRAP / "backend.hcl.example").is_file()
    assert (DEMO / "README.md").is_file()


def test_terraform_centralizes_default_region_and_validates_rds_version() -> None:
    text = _text(DEMO)

    assert 'variable "aws_region"' in text
    assert 'default     = "eu-central-1"' in text or 'default = "eu-central-1"' in text
    assert "region = var.aws_region" in text
    assert 'data "aws_rds_engine_version"' in text
    assert 'preferred_major_targets = ["16"]' in text
    assert "engine_version" in text and "data.aws_rds_engine_version.postgres.version" in text
    assert 'availability_zones = data.aws_availability_zones.available.names' in text


def test_terraform_contains_private_demo_resources_and_https_alb() -> None:
    text = _text(DEMO)

    for resource in (
        'resource "aws_vpc"',
        'resource "aws_ecs_cluster"',
        'resource "aws_ecs_service"',
        'resource "aws_lb"',
        'resource "aws_db_instance"',
        'resource "aws_s3_bucket"',
        'resource "aws_cognito_user_pool"',
        'resource "aws_secretsmanager_secret"',
        'resource "aws_cloudwatch_log_group"',
    ):
        assert resource in text
    assert "publicly_accessible" in text and "false" in text
    assert "block_public_acls       = true" in text
    assert 'protocol' in text and '"HTTPS"' in text
    assert "stickiness" in text
    assert "deployment_circuit_breaker" in text


def test_bootstrap_uses_versioned_s3_state_locking_and_github_oidc_ecr_scan() -> None:
    text = _text(BOOTSTRAP)

    assert "versioning" in text
    assert "use_lockfile = true" in text
    assert 'resource "aws_iam_openid_connect_provider"' in text
    assert 'resource "aws_ecr_repository"' in text
    assert "scan_on_push = true" in text
    assert "sts:TagSession" in text
