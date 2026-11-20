"""Validate portfolio and release-readiness invariants without loading models."""

from __future__ import annotations

import json
from pathlib import Path
import re
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
errors: list[str] = []


def require(relative: str) -> None:
    if not (ROOT / relative).exists():
        errors.append(f"Missing required path: {relative}")


def check_markdown_links() -> None:
    """Check repository-relative Markdown links in public documentation."""
    pattern = re.compile(r"!??\[[^\]]*\]\(([^)]+)\)")
    roots = [ROOT / "README.md", ROOT / "docs", ROOT / "config", ROOT / "reports/release"]
    files = [path for root in roots for path in (root.rglob("*.md") if root.is_dir() else [root])]
    for source in files:
        text = source.read_text(encoding="utf-8")
        for raw_target in pattern.findall(text):
            target = raw_target.strip().split(" ", 1)[0].strip("<>")
            if not target or target.startswith(("http://", "https://", "mailto:", "#")):
                continue
            target_path = (source.parent / target.split("#", 1)[0]).resolve()
            if not target_path.exists():
                errors.append(f"Broken relative link in {source.relative_to(ROOT)}: {target}")


def check_mermaid_wrappers() -> None:
    for source in ("system_architecture", "case_data_flow"):
        wrapper = ROOT / f"docs/diagrams/{source}.md"
        mermaid = ROOT / f"docs/diagrams/{source}.mmd"
        if mermaid.exists() and (
            not wrapper.exists() or "```mermaid" not in wrapper.read_text(encoding="utf-8")
        ):
            errors.append(f"Mermaid wrapper is missing a Mermaid code block: {wrapper}")


def check_release_manifest(registry: dict, manifest: dict) -> None:
    if manifest.get("final_evaluation_path") != "reports/evaluation/phase4/final_complete":
        errors.append("Release manifest does not point to final_complete evaluation artifacts.")
    components = registry.get("components", {})
    for name, component in components.items():
        if name not in manifest.get("canonical_hashes", {}):
            continue
        if manifest["canonical_hashes"][name] != component.get("sha256"):
            errors.append(f"Release manifest hash disagrees with baseline registry: {name}")


def main() -> int:
    for path in (
        "README.md",
        "VERSION",
        "CHANGELOG.md",
        "config/repository_catalog.yaml",
        "config/baseline_registry.json",
        "reports/evaluation/phase4/final_complete/run_manifest.json",
        "reports/evaluation/phase4/final_complete/claim_evidence_index.csv",
        "docs/privacy/PRIVACY_AND_DATA_RETENTION.md",
        "docs/clinical_review/PHASE_4_CLINICAL_SAFETY_REVIEW.md",
        "docs/diagrams/system_architecture.mmd",
        "docs/diagrams/case_data_flow.mmd",
    ):
        require(path)
    for path in (
        "docs/model_cards/face.md",
        "docs/model_cards/speech.md",
        "docs/model_cards/metadata.md",
        "docs/model_cards/acute_symptoms.md",
        "docs/dataset_cards/face.md",
        "docs/dataset_cards/speech.md",
        "docs/dataset_cards/metadata.md",
        "docs/dataset_cards/fer2013.md",
        "docs/dataset_cards/facial_droop_paralysis.md",
        "docs/experiments/EXPERIMENT_INDEX.md",
        "docs/reproducibility/REPRODUCIBILITY.md",
        "docs/testing/TESTING.md",
        "docs/demo/SHOT_LIST.md",
        "docs/release/RELEASE_CHECKLIST.md",
        "reports/release/release_manifest.json",
    ):
        require(path)
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    headings = [
        "## Overview",
        "## Key capabilities",
        "## Architecture",
        "## Model profiles",
        "## Evaluation highlights",
        "## Working application",
        "## Quick start",
        "## Testing and reproducibility",
        "## Limitations and responsible use",
    ]
    positions = [readme.find(item) for item in headings]
    if any(position < 0 for position in positions) or positions != sorted(positions):
        errors.append("README current narrative headings are missing or out of order.")
    try:
        final_root = ROOT / "reports/evaluation/phase4/final_complete"
        face = json.loads((final_root / "face.json").read_text(encoding="utf-8"))
        bundles = json.loads(
            (ROOT / "reports/production/stage5/bundle_benchmarks.json").read_text(encoding="utf-8")
        )
        by_name = {item["bundle"]: item for item in bundles["bundles"]}
        baseline = by_name["baseline"]
        optimized = by_name["mixed-litert"]
        expected_results = (
            f"{face['direct']['roc_auc']:.4f}",
            f"{face['direct']['pr_auc']:.4f}",
            f"{face['adapter_coverage']['accepted']}/{face['total_held_out']} accepted",
            f"({face['adapter_coverage']['coverage'] * 100:.2f}%)",
            f"{baseline['artifact_bytes'] / 1_000_000:.2f} MB",
            f"{optimized['artifact_bytes'] / 1_000_000:.2f} MB",
            f"{baseline['warm_p50_assessment_ms']:.1f} ms",
            f"{optimized['warm_p50_assessment_ms']:.1f} ms",
        )
        for value in expected_results:
            if value not in readme:
                errors.append(f"README result is not synchronized with its source report: {value}")
    except (OSError, KeyError, TypeError, ValueError):
        errors.append("Could not validate README values against Phase 4 and Stage 5 artifacts.")
    for required_scope in (
        "not a diagnostic system",
        "not a native smartphone benchmark",
        "not a claim of IEEE Xplore indexing",
        "no prospective rural or external clinical validation",
    ):
        if required_scope.casefold() not in readme.casefold():
            errors.append(f"README is missing required scope qualification: {required_scope}")
    if re.search(r"[A-Za-z]:\\|/Users/|/home/|C:/Users/", readme):
        errors.append("README contains an absolute local path.")
    if re.search(r"\b(TODO|TBD|PLACEHOLDER|Lorem ipsum)\b", readme, re.I):
        errors.append("README contains a placeholder.")
    public_files = [
        path
        for folder in (ROOT / "docs", ROOT / "config", ROOT / "reports/release")
        for path in folder.rglob("*.md")
        if folder.exists()
    ] + [ROOT / "README.md", ROOT / "CHANGELOG.md"]
    for path in public_files:
        text = path.read_text(encoding="utf-8")
        if re.search(r"[A-Za-z]:\\|/Users/|/home/|C:/Users/", text):
            errors.append(f"Absolute local path found in {path.relative_to(ROOT)}")
        if re.search(r"\b(TODO|TBD|FIXME|PLACEHOLDER|Lorem ipsum)\b", text, re.I):
            errors.append(f"Unresolved placeholder found in {path.relative_to(ROOT)}")
    for card in (ROOT / "docs/model_cards").glob("*.md"):
        text = card.read_text(encoding="utf-8")
        for heading in ("## Intended use", "## Out of scope", "## Limitations"):
            if heading not in text:
                errors.append(f"Model card missing {heading}: {card}")
    for card in (ROOT / "docs/dataset_cards").glob("*.md"):
        if "Limitation:" not in card.read_text(encoding="utf-8"):
            errors.append(f"Dataset card missing limitation: {card}")
    version = (ROOT / "VERSION").read_text(encoding="utf-8").strip()
    try:
        registry = json.loads((ROOT / "config/baseline_registry.json").read_text(encoding="utf-8"))
        pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
        release_manifest = json.loads(
            (ROOT / "reports/release/release_manifest.json").read_text(encoding="utf-8")
        )
        if f'version = "{version}"' not in pyproject or release_manifest.get("version") != version:
            errors.append(
                "Version mismatch between VERSION, pyproject.toml, and release manifest."
            )
        check_release_manifest(registry, release_manifest)
    except (OSError, ValueError, TypeError):
        errors.append("Release manifest or version could not be parsed.")
    shot_list = ROOT / "docs/demo/SHOT_LIST.md"
    if shot_list.exists() and "PENDING" not in shot_list.read_text(encoding="utf-8"):
        errors.append("Demo media status is not explicit.")
    catalog_path = ROOT / "config/repository_catalog.yaml"
    if catalog_path.exists():
        catalog = yaml.safe_load(catalog_path.read_text(encoding="utf-8"))
        for item in catalog.get("categories", {}).get("ACTIVE", []):
            if not (ROOT / item["path"]).exists():
                errors.append(f"Catalog ACTIVE path missing: {item['path']}")
    claims = ROOT / "reports/evaluation/phase4/final_complete/claim_evidence_index.csv"
    if claims.exists():
        for line in claims.read_text(encoding="utf-8").splitlines()[1:]:
            fields = line.split(",")
            if len(fields) >= 3 and fields[1] == "supported":
                for evidence in fields[2].strip('"').split("; "):
                    if evidence and not (claims.parent / evidence).exists():
                        errors.append(f"Supported claim evidence missing: {evidence}")
    for path in (ROOT / "rural_stroke_assist" / "assessment").glob("*.py"):
        if "fusion_engine" in path.read_text(encoding="utf-8"):
            errors.append(f"Active assessment code references legacy fusion: {path}")
    privacy = (
        (ROOT / "docs/privacy/PRIVACY_AND_DATA_RETENTION.md").read_text(encoding="utf-8")
        if (ROOT / "docs/privacy/PRIVACY_AND_DATA_RETENTION.md").exists()
        else ""
    )
    if "pseudonymous" not in privacy.lower() or "--confirm" not in privacy:
        errors.append("Privacy documentation is incomplete.")
    check_markdown_links()
    check_mermaid_wrappers()
    if errors:
        print("RELEASE READINESS: FAIL")
        print("\n".join(f"- {error}" for error in errors))
        return 1
    print("RELEASE READINESS: PASS")
    print(
        "Validated catalog, current README structure and source metrics, supported claim evidence, privacy wording, and active fusion boundary."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
