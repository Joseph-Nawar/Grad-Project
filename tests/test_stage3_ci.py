from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).parents[1]


def test_pr_ci_covers_python_iac_images_scanning_compose_and_attestations() -> None:
    workflow = (ROOT / ".github/workflows/ci-stage3.yml").read_text(encoding="utf-8")

    for required in ("pytest", "terraform fmt", "terraform validate", "docker build", "aquasec/trivy@sha256:", "anchore/syft@sha256:", "--provenance", "scripts/stack.py up"):
        assert required in workflow
    assert "CRITICAL" in workflow


def test_demo_cd_is_controlled_and_promotes_exact_digests_after_migration() -> None:
    workflow = (ROOT / ".github/workflows/deploy-demo.yml").read_text(encoding="utf-8")

    assert "workflow_dispatch" in workflow
    assert "environment: stage3-demo" in workflow
    assert "aws-actions/configure-aws-credentials" in workflow
    assert "aws ecs run-task" in workflow
    assert "migration_task_definition" in workflow
    assert "aws ecs update-service" in workflow
    assert "image_digest" in workflow
    assert "aws ecs wait services-stable" in workflow
    assert "https://" in workflow
