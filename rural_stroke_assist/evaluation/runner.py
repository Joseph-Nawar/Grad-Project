"""Phase 4 evaluation orchestration."""
from __future__ import annotations
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import importlib.metadata
import json
import platform
import uuid

from rural_stroke_assist.inference.registry import load_baseline_registry
from .ablation import run_weight_sensitivity
from .artifact_writer import prepare_output, write_json
from .claim_registry import write_claim_index
from .config import EvaluationConfig, load_evaluation_config
from .modality_evaluators import evaluate_face, evaluate_metadata, evaluate_speech
from .report_builder import write_report
from .robustness import scenario_coverage
from .symptom_evaluator import evaluate_symptoms
from .profiling import profile_default_assessment
from .gap_closure import face_coverage_audit, speech_speaker_audit, corruption_audit, missing_modality_audit, stability_audit, cold_start_audit, metadata_version_comparison


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""): digest.update(block)
    return digest.hexdigest()


def run_evaluation(suite: str, *, config_path: str | Path = "config/evaluation/phase4.yaml", output_dir: str | Path | None = None, overwrite: bool = False, sklearn_142_python: str | None = None) -> Path:
    config = load_evaluation_config(config_path); registry = load_baseline_registry(); run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    destination = Path(output_dir) if output_dir else Path(config.output_root) / run_id
    destination = prepare_output(destination, overwrite=overwrite)
    iterations = config.smoke_bootstrap_iterations if suite == "smoke" else config.bootstrap_iterations
    limit = 5 if suite == "smoke" else None
    results: dict[str, object] = {}; warnings: list[str] = []
    if suite in {"smoke", "modality", "full"}:
        results["face"] = evaluate_face(config.face_manifest, registry, limit=limit, bootstrap_iterations=iterations, seed=config.seed, output_dir=destination / "plots")
        results["speech"] = evaluate_speech(config.speech_manifest, registry, limit=limit, bootstrap_iterations=iterations, seed=config.seed, output_dir=destination / "plots")
        results["metadata_context"] = evaluate_metadata(config.metadata_manifest, registry, limit=limit, bootstrap_iterations=iterations, seed=config.seed, output_dir=destination / "plots")
    if suite in {"smoke", "system", "full"}:
        results["symptoms"] = evaluate_symptoms(); results["robustness"] = scenario_coverage(); results["ablations"] = run_weight_sensitivity()
    if suite in {"system", "full"}:
        results["profiling"] = profile_default_assessment(warm_runs=config.warm_runs if suite == "full" else 3, cold_runs=config.cold_runs if suite == "full" else 1)
    if suite == "full":
        results["face_coverage"] = face_coverage_audit(config.face_manifest, registry, destination)
        results["speech_speakers"] = speech_speaker_audit(config.speech_manifest, registry, destination)
        results["corruption"] = corruption_audit(destination)
        results["missing_modalities"] = missing_modality_audit(destination, registry)
        results["stability"] = stability_audit(destination)
        results["cold_start"] = cold_start_audit(destination, runs=config.cold_runs)
        results["metadata_version_comparison"] = metadata_version_comparison(destination, sklearn_142_python)
    hashes = {name: registry.component(name).data.get("sha256") for name in ("face", "speech", "metadata_context", "acute_symptoms", "fusion")}
    manifest_hashes = {name: _sha(registry.manifest_path(name)) for name in ("face_split", "speech_split", "metadata_split")}
    config_hash = _sha(Path(config_path))
    manifest = {"run_id": run_id, "utc_started": datetime.now(timezone.utc).isoformat(), "suite": suite, "command": "python scripts/run_phase4_evaluation.py", "config": str(config_path), "config_sha256": config_hash, "seed": config.seed, "iterations": iterations, "platform": {"python": platform.python_version(), "system": platform.platform()}, "dependencies": {name: importlib.metadata.version(name) for name in ("numpy", "pandas", "scikit-learn", "tensorflow", "joblib", "psutil") if _installed(name)}, "canonical_hashes": hashes, "manifest_hashes": manifest_hashes, "warnings": ["Metadata artifact was serialized with scikit-learn 1.4.2 and loaded in the validated 1.5.2 runtime."] if suite in {"modality", "full", "smoke"} else [], "generated_artifacts": []}
    write_json(destination / "run_manifest.json", manifest)
    for key, value in results.items(): write_json(destination / f"{key}.json", value)
    write_claim_index(destination / "claim_evidence_index.csv")
    write_report(destination / "PHASE_4_EVALUATION_REPORT.md", run_id=run_id, suite=suite, results=results, warnings=warnings)
    manifest["generated_artifacts"] = sorted(str(path.relative_to(destination)) for path in destination.rglob("*") if path.is_file() and path.name != "run_manifest.json")
    write_json(destination / "run_manifest.json", manifest)
    return destination


def _installed(name: str) -> bool:
    try: importlib.metadata.version(name); return True
    except importlib.metadata.PackageNotFoundError: return False
