from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rural_stroke_assist.modules.fusion_module import (
    DEFAULT_FUSION_WEIGHTS, DEFAULT_HIGH_THRESHOLD, DEFAULT_MODERATE_THRESHOLD,
    FusionInput, fuse_multimodal_scores,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_DIR = REPO_ROOT / "reports/experiments/pretrained_reference_profile_compatibility_audit"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "reports/experiments/fusion_weight_sensitivity"
DEFAULT_REPORT_PATH = REPO_ROOT.parent / "codex_workplace" / REPO_ROOT.name / "experiment_reports/fusion_weight_sensitivity/REPORT.md"
WEIGHT_NAMES = ("face", "speech", "acute_symptoms", "metadata_context")
SCORE_FIELDS = {"face": "face_acute_score", "speech": "speech_acute_score", "acute_symptoms": "acute_symptom_score", "metadata_context": "metadata_contextual_risk_score"}
BASELINE_WEIGHTS = {"face": 0.35, "speech": 0.30, "acute_symptoms": 0.25, "metadata_context": 0.10}
PROFILE_MAPPING = {"canonical": "edge", "pretrained": "reference"}
EXPECTED_HAND_SCENARIOS = {
    "all_evidence_low", "face_high_others_low", "speech_high_others_low", "metadata_high_acute_low",
    "face_speech_corroborating_high", "strong_acute_symptoms_learned_low", "conflicting_face_and_speech",
    "missing_face", "missing_speech", "missing_metadata", "symptoms_only", "no_usable_evidence",
}
RISK_BANDS = ("LOW", "MODERATE", "HIGH", "URGENT")
MARGIN_BINS = ("[0.000, 0.025)", "[0.025, 0.050)", "[0.050, 0.100)", "[0.100, infinity)")
SOURCE_FILENAMES = ("fusion_scenario_comparison.csv", "distribution_grounded_scenarios.csv", "provenance_manifest.json")
HASHED_INPUTS = ("rural_stroke_assist/modules/fusion_module.py", "config/baseline_registry.json")


@dataclass(frozen=True)
class WeightVector:
    vector_id: str
    vector_set: str
    perturbation_class: str
    weights: Mapping[str, float]
    max_abs_delta: float
    target_modality: str | None = None
    target_delta: float | None = None


@dataclass(frozen=True)
class Scenario:
    profile: str
    source_profile: str
    scenario_family: str
    name: str
    scores: Mapping[str, float | None]
    hard_escalation: bool
    source_status: str
    source_baseline_score: float | None
    source_baseline_band: str | None
    source_artifact: str
    availability_mask: str

    @property
    def available_modalities(self) -> tuple[str, ...]:
        return tuple(name for name in WEIGHT_NAMES if self.scores.get(name) is not None)

    @property
    def available_modality_count(self) -> int:
        return len(self.available_modalities)

    @property
    def scenario_id(self) -> str:
        return f"{self.scenario_family}|{self.profile}|{self.name}"


class PolicyMismatchError(ValueError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return value


def load_frozen_policy(repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    path = repo_root / "config/baseline_registry.json"
    if not path.is_file():
        raise FileNotFoundError(f"Required baseline registry is missing: {path}")
    registry = _read_json(path)
    fusion = registry.get("fusion")
    if not isinstance(fusion, dict) or not isinstance(fusion.get("weights"), dict):
        raise PolicyMismatchError("Baseline registry is missing its fusion weights.")
    raw = fusion["weights"]
    weights = {name: float(raw.get(name, math.nan)) for name in WEIGHT_NAMES}
    if set(raw) != set(WEIGHT_NAMES) or weights != BASELINE_WEIGHTS:
        raise PolicyMismatchError(
            "Expected frozen weights face=0.35, speech=0.30, acute_symptoms=0.25, "
            f"metadata_context=0.10; found {raw!r}."
        )
    if not math.isclose(sum(weights.values()), 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise PolicyMismatchError("Baseline weights do not sum to 1.0.")
    moderate = float(fusion.get("moderate_threshold", math.nan))
    high = float(fusion.get("high_threshold", math.nan))
    floor = float(fusion.get("hard_escalation_score_floor", math.nan))
    if (moderate, high, floor) != (0.50, 0.75, 0.85):
        raise PolicyMismatchError(
            f"Expected thresholds (0.50, 0.75) and hard floor 0.85; found {(moderate, high, floor)}."
        )
    if dict(DEFAULT_FUSION_WEIGHTS) != weights:
        raise PolicyMismatchError(f"Fusion defaults {DEFAULT_FUSION_WEIGHTS!r} disagree with registry {weights!r}.")
    if DEFAULT_MODERATE_THRESHOLD != moderate or DEFAULT_HIGH_THRESHOLD != high:
        raise PolicyMismatchError("Fusion module thresholds disagree with baseline registry.")
    return {"baseline_weights": weights, "moderate_threshold": moderate, "high_threshold": high,
            "hard_escalation_score_floor": floor, "registry_path": path}


def _validate_weights(weights: Mapping[str, float], label: str) -> None:
    if set(weights) != set(WEIGHT_NAMES):
        raise ValueError(f"{label}: expected exactly {WEIGHT_NAMES}, got {tuple(weights)}.")
    values = [float(weights[name]) for name in WEIGHT_NAMES]
    if any(not math.isfinite(value) or value < 0.0 for value in values):
        raise ValueError(f"{label}: weights must be finite and non-negative: {weights!r}.")
    if not math.isclose(sum(values), 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(f"{label}: weights do not sum to 1.0: {weights!r}.")


def generate_simplex_vectors(max_deviation: float = 0.10) -> list[WeightVector]:
    if max_deviation not in (0.05, 0.10):
        raise ValueError("Simplex envelope must be 0.05 or 0.10.")
    units = int(round(max_deviation / 0.05))
    base = tuple(round(BASELINE_WEIGHTS[name] / 0.05) for name in WEIGHT_NAMES)
    grid = [tuple(v) for v in itertools.product(range(21), repeat=4)
            if sum(v) == 20 and all(abs(v[i] - base[i]) <= units for i in range(4))]
    grid.sort(key=lambda v: (v != base, v))
    vectors = []
    for index, point in enumerate(grid):
        weights = {name: point[i] * 0.05 for i, name in enumerate(WEIGHT_NAMES)}
        _validate_weights(weights, f"simplex vector {point}")
        shift = max(abs(point[i] - base[i]) * 0.05 for i in range(4))
        kind = "baseline" if point == base else "local_5pp" if shift <= 0.05 else "stress_5_to_10pp"
        vectors.append(WeightVector(f"SW-{index:03d}", "simplex", kind, weights, shift))
    expected = 19 if max_deviation == 0.05 else 85
    if len(vectors) != expected:
        raise AssertionError(f"Simplex grid count {len(vectors)} differs from expected {expected}.")
    return vectors


def generate_targeted_vectors() -> list[WeightVector]:
    vectors = []
    for modality in WEIGHT_NAMES:
        base = BASELINE_WEIGHTS[modality]
        for delta in (-0.10, -0.05, 0.05, 0.10):
            weights = {
                name: (base + delta if name == modality else BASELINE_WEIGHTS[name] * (1 - delta / (1 - base)))
                for name in WEIGHT_NAMES
            }
            _validate_weights(weights, f"targeted {modality} {delta:+.2f}")
            sign = "p" if delta > 0 else "m"
            amount = int(round(abs(delta) * 100))
            shift = max(abs(weights[name] - BASELINE_WEIGHTS[name]) for name in WEIGHT_NAMES)
            vectors.append(WeightVector(f"TW-{modality}-{sign}{amount:02d}", "targeted",
                                        "targeted_modality", weights, shift, modality, delta))
    if len(vectors) != 16:
        raise AssertionError(f"Targeted vector count {len(vectors)} differs from expected 16.")
    return vectors


def _optional_float(value: str | None, field: str, source: str) -> float | None:
    if value is None or not value.strip():
        return None
    number = float(value)
    if not math.isfinite(number) or not 0 <= number <= 1:
        raise ValueError(f"{source}: {field} is outside [0, 1]: {value!r}.")
    return number


def _make_scenario(row: Mapping[str, str], family: str, artifact: str) -> Scenario:
    source_profile = row.get("profile", "")
    if source_profile not in PROFILE_MAPPING:
        raise ValueError(f"Unexpected source profile {source_profile!r} in {artifact}.")
    scores = {name: _optional_float(row.get(field), field, artifact) for name, field in SCORE_FIELDS.items()}
    status = row.get("status", "").strip()
    count = sum(value is not None for value in scores.values())
    if (count == 0 and status != "no_usable_evidence") or (count > 0 and status != "completed"):
        raise ValueError(f"Unexpected source status {status!r} for {row.get('scenario_name')!r}.")
    available = tuple(name for name in WEIGHT_NAMES if scores[name] is not None)
    stored = tuple(name for name in (row.get("available_modalities") or "").split(",") if name)
    if available != stored:
        raise ValueError(f"Availability mismatch for {row.get('scenario_name')!r}: {available} != {stored}.")
    mask = "".join("1" if scores[name] is not None else "0" for name in WEIGHT_NAMES)
    return Scenario(
        PROFILE_MAPPING[source_profile], source_profile, family, row["scenario_name"], scores,
        row.get("acute_symptom_hard_escalation", "false").lower() == "true", status,
        _optional_float(row.get("fused_score"), "fused_score", artifact),
        (row.get("risk_band") or "").strip() or None, artifact, mask,
    )


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def load_compatibility_sources(source_dir: Path = DEFAULT_SOURCE_DIR) -> dict[str, Any]:
    paths = {name: source_dir / name for name in SOURCE_FILENAMES}
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Required compatibility artifact(s) missing: " + ", ".join(missing))
    provenance = _read_json(paths["provenance_manifest.json"])
    if provenance.get("test_rows_used") is not False or provenance.get("test_results_used_for_tuning") is not False:
        raise ValueError("Compatibility provenance does not confirm held-out test rows/results were unused.")
    comparison = _read_csv(paths["fusion_scenario_comparison.csv"])
    quantiles = _read_csv(paths["distribution_grounded_scenarios.csv"])
    hand = [_make_scenario(row, "hand_authored", "reports/experiments/pretrained_reference_profile_compatibility_audit/fusion_scenario_comparison.csv")
            for row in comparison if row.get("scenario_type") == "hand_authored"]
    distribution = [_make_scenario(row, "distribution_grounded", "reports/experiments/pretrained_reference_profile_compatibility_audit/distribution_grounded_scenarios.csv")
                    for row in quantiles if row.get("scenario_type") == "distribution_grounded"]
    for profile in PROFILE_MAPPING.values():
        h = [item for item in hand if item.profile == profile]
        names = {item.name for item in h}
        if len(h) != 12 or names != EXPECTED_HAND_SCENARIOS:
            raise ValueError(f"{profile} hand-authored scenarios differ from required 12: {sorted(names)}")
        q = [item for item in distribution if item.profile == profile]
        if len(q) != 16 or len({item.name for item in q}) != 16:
            raise ValueError(f"{profile} must have exactly 16 stored distribution scenarios.")
    return {"hand_authored": hand, "distribution_grounded": distribution,
            "profile_mapping": dict(PROFILE_MAPPING), "provenance": provenance,
            "loaded_paths": [paths[name] for name in SOURCE_FILENAMES]}


def build_availability_mask_scenarios(conflict: Scenario) -> list[Scenario]:
    if conflict.name != "conflicting_face_and_speech" or conflict.available_modality_count != 4:
        raise ValueError("Availability masks require the complete conflict evidence tuple.")
    cases = []
    for bits in itertools.product((0, 1), repeat=4):
        mask = "".join(map(str, bits))
        scores = {name: conflict.scores[name] if bits[i] else None for i, name in enumerate(WEIGHT_NAMES)}
        count = sum(bits)
        cases.append(Scenario(conflict.profile, conflict.source_profile, "availability_mask",
                              f"conflicting_face_and_speech__mask_{mask}", scores,
                              conflict.hard_escalation, "completed" if count else "no_usable_evidence",
                              None, None, conflict.source_artifact, mask))
    if len(cases) != 16:
        raise AssertionError("Availability masks must include exactly 16 cases.")
    return cases


def build_all_scenarios(source: Mapping[str, Any]) -> list[Scenario]:
    scenarios = [*source["hand_authored"], *source["distribution_grounded"]]
    for profile in PROFILE_MAPPING.values():
        conflict = next(item for item in source["hand_authored"]
                        if item.name == "conflicting_face_and_speech" and item.profile == profile)
        scenarios.extend(build_availability_mask_scenarios(conflict))
    counts = Counter(item.scenario_family for item in scenarios)
    if dict(counts) != {"hand_authored": 24, "distribution_grounded": 32, "availability_mask": 32}:
        raise AssertionError(f"Unexpected scenario family counts: {dict(counts)}.")
    return scenarios


def evaluate_scenario(scenario: Scenario, weights: Mapping[str, float], policy: Mapping[str, Any]) -> dict[str, Any]:
    _validate_weights(weights, f"{scenario.scenario_id} weights")
    if scenario.available_modality_count == 0:
        return {"status": "no_usable_evidence", "fused_score": None, "risk_band": None,
                "effective_weights": {}, "error_type": None, "error_message": None}
    fusion_input = FusionInput(
        face_acute_score=scenario.scores["face"], speech_acute_score=scenario.scores["speech"],
        acute_symptom_score=scenario.scores["acute_symptoms"],
        metadata_contextual_risk_score=scenario.scores["metadata_context"],
        acute_symptom_hard_escalation=scenario.hard_escalation,
    )
    try:
        result = fuse_multimodal_scores(fusion_input, weights=dict(weights),
                                        moderate_threshold=policy["moderate_threshold"],
                                        high_threshold=policy["high_threshold"])
    except ValueError as exc:
        if str(exc) != "At least one modality score must be available.":
            raise
        return {"status": "fusion_error", "fused_score": None, "risk_band": None,
                "effective_weights": {}, "error_type": type(exc).__name__, "error_message": str(exc)}
    score = float(result.fused_score)
    if not math.isfinite(score) or not 0 <= score <= 1:
        raise AssertionError(f"Fusion returned non-finite or unbounded score for {scenario.scenario_id}: {score}.")
    effective = dict(result.normalized_weights_used)
    if not math.isclose(sum(effective.values()), 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise AssertionError(f"Effective weights do not sum to one for {scenario.scenario_id}.")
    return {"status": "completed", "fused_score": score, "risk_band": result.risk_band,
            "effective_weights": effective, "error_type": None, "error_message": None}


def transition_label(baseline_band: str | None, perturbed_band: str | None) -> str | None:
    return None if baseline_band is None or perturbed_band is None else f"{baseline_band}->{perturbed_band}"


def _vector_record(vector: WeightVector) -> dict[str, Any]:
    row: dict[str, Any] = {
        "weight_vector_id": vector.vector_id, "vector_set": vector.vector_set,
        "perturbation_class": vector.perturbation_class, "target_modality": vector.target_modality,
        "target_delta": vector.target_delta, "max_abs_delta_from_baseline": vector.max_abs_delta,
    }
    row.update({f"weight_{name}": vector.weights[name] for name in WEIGHT_NAMES})
    row["weights_json"] = dict(vector.weights)
    return row


def _scenario_record(item: Scenario) -> dict[str, Any]:
    row = {
        "scenario_id": item.scenario_id, "profile": item.profile, "source_profile": item.source_profile,
        "scenario_family": item.scenario_family, "scenario_name": item.name,
        "source_artifact": item.source_artifact, "availability_mask": item.availability_mask,
        "available_modalities": ",".join(item.available_modalities),
        "available_modality_count": item.available_modality_count,
        "acute_symptom_hard_escalation": item.hard_escalation, "source_status": item.source_status,
        "source_baseline_fused_score": item.source_baseline_score,
        "source_baseline_risk_band": item.source_baseline_band,
    }
    row.update({f"score_{name}": item.scores[name] for name in WEIGHT_NAMES})
    return row


def _comparison_record(scenario: Scenario, baseline: Mapping[str, Any], perturbed: Mapping[str, Any],
                       vector: WeightVector, base_weights: Mapping[str, float]) -> dict[str, Any]:
    base_score, changed_score = baseline["fused_score"], perturbed["fused_score"]
    delta = None if base_score is None or changed_score is None else changed_score - base_score
    row: dict[str, Any] = {
        "scenario_id": scenario.scenario_id, "profile": scenario.profile,
        "source_profile": scenario.source_profile, "scenario_family": scenario.scenario_family,
        "scenario_name": scenario.name, "availability_mask": scenario.availability_mask,
        "available_modalities": ",".join(scenario.available_modalities),
        "available_modality_count": scenario.available_modality_count,
        "acute_symptom_hard_escalation": scenario.hard_escalation,
        "weight_vector_id": vector.vector_id, "vector_set": vector.vector_set,
        "perturbation_class": vector.perturbation_class, "target_modality": vector.target_modality,
        "target_delta": vector.target_delta, "baseline_weights_json": dict(base_weights),
        "perturbed_weights_json": dict(vector.weights),
        "baseline_effective_weights_json": dict(baseline["effective_weights"]),
        "perturbed_effective_weights_json": dict(perturbed["effective_weights"]),
        "baseline_status": baseline["status"], "perturbed_status": perturbed["status"],
        "baseline_fused_score": base_score, "perturbed_fused_score": changed_score,
        "signed_score_delta": delta, "absolute_score_delta": None if delta is None else abs(delta),
        "baseline_risk_band": baseline["risk_band"], "perturbed_risk_band": perturbed["risk_band"],
        "band_changed": None if baseline["risk_band"] is None or perturbed["risk_band"] is None else baseline["risk_band"] != perturbed["risk_band"],
        "transition_label": transition_label(baseline["risk_band"], perturbed["risk_band"]),
        "fusion_error_type": perturbed["error_type"], "fusion_error_message": perturbed["error_message"],
    }
    for name in WEIGHT_NAMES:
        row[f"baseline_weight_{name}"] = base_weights[name]
        row[f"weight_{name}"] = vector.weights[name]
        row[f"baseline_effective_weight_{name}"] = baseline["effective_weights"].get(name)
        row[f"effective_weight_{name}"] = perturbed["effective_weights"].get(name)
    return row


def _assert_source_baseline_reproduced(item: Scenario, baseline: Mapping[str, Any]) -> None:
    if item.scenario_family == "availability_mask":
        return
    if item.source_status != baseline["status"]:
        raise AssertionError(f"Source status replay mismatch for {item.scenario_id}.")
    if item.source_baseline_score is None:
        if baseline["fused_score"] is not None:
            raise AssertionError(f"Empty source scenario acquired a score: {item.scenario_id}.")
    elif baseline["fused_score"] is None or not math.isclose(
        item.source_baseline_score, baseline["fused_score"], rel_tol=0.0, abs_tol=1e-10
    ):
        raise AssertionError(
            f"Baseline replay differs for {item.scenario_id}: source={item.source_baseline_score}, "
            f"replay={baseline['fused_score']}."
        )
    if item.source_baseline_band != baseline["risk_band"]:
        raise AssertionError(f"Source risk band replay mismatch for {item.scenario_id}.")


def build_sensitivity_results(scenarios: Sequence[Scenario], simplex: Sequence[WeightVector],
                              targeted: Sequence[WeightVector], policy: Mapping[str, Any]
                              ) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    baseline_vector = next(item for item in simplex if item.perturbation_class == "baseline")
    baselines: dict[str, dict[str, Any]] = {}
    rows = []
    for scenario in scenarios:
        baseline = evaluate_scenario(scenario, baseline_vector.weights, policy)
        _assert_source_baseline_reproduced(scenario, baseline)
        baselines[scenario.scenario_id] = baseline
        for vector in simplex:
            perturbed = baseline if vector.vector_id == baseline_vector.vector_id else evaluate_scenario(scenario, vector.weights, policy)
            rows.append(_comparison_record(scenario, baseline, perturbed, vector, policy["baseline_weights"]))
        for vector in targeted:
            perturbed = evaluate_scenario(scenario, vector.weights, policy)
            rows.append(_comparison_record(scenario, baseline, perturbed, vector, policy["baseline_weights"]))
    expected = len(scenarios) * (len(simplex) + len(targeted))
    if len(rows) != expected:
        raise AssertionError(f"Sensitivity row count {len(rows)} differs from expected {expected}.")
    return rows, baselines


def _aggregate(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    comparable = [row for row in rows if row["baseline_status"] == "completed" and row["perturbed_status"] == "completed" and row["baseline_risk_band"] is not None and row["perturbed_risk_band"] is not None]
    shifts = [float(row["absolute_score_delta"]) for row in rows if row["absolute_score_delta"] is not None]
    stable = sum(row["band_changed"] is False for row in comparable)
    stability = stable / len(comparable) if comparable else None
    return {
        "perturbation_comparisons": len(rows), "completed_band_comparisons": len(comparable),
        "score_comparisons": len(shifts),
        "fusion_error_count": sum(row["perturbed_status"] == "fusion_error" for row in rows),
        "no_usable_evidence_count": sum(row["perturbed_status"] == "no_usable_evidence" for row in rows),
        "band_stability": stability, "band_flip_rate": None if stability is None else 1.0 - stability,
        "mean_absolute_score_change": statistics.fmean(shifts) if shifts else None,
        "median_absolute_score_change": statistics.median(shifts) if shifts else None,
        "maximum_absolute_score_change": max(shifts) if shifts else None,
    }


def _groups(rows: Sequence[Mapping[str, Any]], keys: Sequence[str]) -> dict[tuple[Any, ...], list[Mapping[str, Any]]]:
    grouped: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row.get(key) for key in keys)].append(row)
    return grouped


def build_stability_summary(results: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    primary = [row for row in results if row["vector_set"] == "simplex" and row["perturbation_class"] != "baseline"]
    scopes = (
        ("overall", ()), ("profile", ("profile",)), ("scenario_family", ("scenario_family",)),
        ("availability_count", ("available_modality_count",)),
        ("profile_and_family", ("profile", "scenario_family")),
        ("profile_and_availability", ("profile", "available_modality_count")),
        ("family_and_availability", ("scenario_family", "available_modality_count")),
        ("profile_family_availability", ("profile", "scenario_family", "available_modality_count")),
    )
    output = []
    for scope, dimensions in scopes:
        keys = (*dimensions, "perturbation_class")
        for values, rows in _groups(primary, keys).items():
            selected = dict(zip(keys, values))
            record = {
                "grouping": scope,
                "group_value": "all" if not dimensions else " | ".join(f"{key}={selected[key]}" for key in dimensions),
                "profile": selected.get("profile", "all"), "scenario_family": selected.get("scenario_family", "all"),
                "perturbation_class": selected["perturbation_class"],
                "available_modality_count": selected.get("available_modality_count", "all"),
                "baseline_scenario_count": len({row["scenario_id"] for row in rows}),
            }
            record.update(_aggregate(rows))
            output.append(record)
    return output


def build_scenario_sensitivity(results: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    primary = [row for row in results if row["vector_set"] == "simplex" and row["perturbation_class"] != "baseline"]
    keys = ("profile", "scenario_family", "scenario_name", "perturbation_class")
    output = []
    for values, rows in sorted(_groups(primary, keys).items()):
        record = dict(zip(keys, values))
        record["scenario_id"] = rows[0]["scenario_id"]
        record["available_modality_count"] = rows[0]["available_modality_count"]
        record.update(_aggregate(rows))
        output.append(record)
    return output


def _transition_counts(rows: Sequence[Mapping[str, Any]]) -> Counter[tuple[str, str]]:
    counts: Counter[tuple[str, str]] = Counter()
    for row in rows:
        if row["transition_label"] and row["baseline_status"] == "completed" and row["perturbed_status"] == "completed":
            counts[(row["baseline_risk_band"], row["perturbed_risk_band"])] += 1
    return counts


def build_band_transitions(results: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    matrix = tuple(itertools.product(RISK_BANDS, repeat=2))
    primary = [row for row in results if row["vector_set"] == "simplex" and row["perturbation_class"] != "baseline"]
    targeted = [row for row in results if row["vector_set"] == "targeted"]
    specs: list[tuple[str, dict[str, Any], list[Mapping[str, Any]]]] = []
    for kind in ("local_5pp", "stress_5_to_10pp"):
        selected = [row for row in primary if row["perturbation_class"] == kind]
        specs.append(("simplex", {"perturbation_class": kind}, selected))
        for profile in ("edge", "reference"):
            specs.append(("simplex", {"perturbation_class": kind, "profile": profile}, [r for r in selected if r["profile"] == profile]))
        for family in ("hand_authored", "distribution_grounded", "availability_mask"):
            specs.append(("simplex", {"perturbation_class": kind, "scenario_family": family}, [r for r in selected if r["scenario_family"] == family]))
    for modality in WEIGHT_NAMES:
        for delta in (-0.10, -0.05, 0.05, 0.10):
            selected = [r for r in targeted if r["target_modality"] == modality and r["target_delta"] == delta]
            delta_text = f"{delta:+.2f}"
            specs.append(("targeted", {"target_modality": modality, "target_delta": delta_text}, selected))
            for profile in ("edge", "reference"):
                specs.append(("targeted", {"target_modality": modality, "target_delta": delta_text, "profile": profile}, [r for r in selected if r["profile"] == profile]))
    output = []
    for comparison_set, group, rows in specs:
        counts = _transition_counts(rows)
        for baseline, perturbed in matrix:
            output.append({
                "comparison_set": comparison_set,
                "perturbation_class": group.get("perturbation_class", "targeted_modality"),
                "profile": group.get("profile", "all"), "scenario_family": group.get("scenario_family", "all"),
                "target_modality": group.get("target_modality", "all"), "target_delta": group.get("target_delta", "all"),
                "baseline_band": baseline, "perturbed_band": perturbed,
                "transition_label": f"{baseline}->{perturbed}", "count": counts[(baseline, perturbed)],
            })
    return output


def build_targeted_modality_summary(results: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    primary = [row for row in results if row["vector_set"] == "targeted"]
    output = []
    for modality in WEIGHT_NAMES:
        for delta in (-0.10, -0.05, 0.05, 0.10):
            selected = [r for r in primary if r["target_modality"] == modality and r["target_delta"] == delta]
            for profile in ("all", "edge", "reference"):
                rows = selected if profile == "all" else [r for r in selected if r["profile"] == profile]
                record: dict[str, Any] = {"modality": modality, "delta": delta, "profile": profile}
                record.update(_aggregate(rows))
                counts = _transition_counts(rows)
                record["transition_counts_json"] = {f"{a}->{b}": counts[(a, b)] for a, b in itertools.product(RISK_BANDS, repeat=2) if counts[(a, b)]}
                output.append(record)
    return output


def threshold_margin_bin(margin: float) -> str:
    if not math.isfinite(margin) or margin < 0:
        raise ValueError(f"Invalid threshold margin {margin!r}.")
    if margin < 0.025: return MARGIN_BINS[0]
    if margin < 0.050: return MARGIN_BINS[1]
    if margin < 0.100: return MARGIN_BINS[2]
    return MARGIN_BINS[3]


def build_threshold_margin_analysis(results: Sequence[Mapping[str, Any]], baselines: Mapping[str, Mapping[str, Any]],
                                    scenarios: Sequence[Scenario], policy: Mapping[str, Any]) -> list[dict[str, Any]]:
    scenario_by_id = {item.scenario_id: item for item in scenarios}
    margins = {}
    for scenario_id, baseline in baselines.items():
        scenario = scenario_by_id[scenario_id]
        if scenario.hard_escalation or baseline["status"] != "completed" or baseline["fused_score"] is None:
            continue
        score = float(baseline["fused_score"])
        margin = min(abs(score - policy["moderate_threshold"]), abs(score - policy["high_threshold"]))
        margins[scenario_id] = threshold_margin_bin(margin)
    candidates = [r for r in results if r["vector_set"] == "simplex" and r["perturbation_class"] != "baseline" and r["scenario_id"] in margins]
    output = []
    for margin_bin in MARGIN_BINS:
        for profile in ("all", "edge", "reference"):
            for kind in ("all", "local_5pp", "stress_5_to_10pp"):
                scenario_ids = [sid for sid, value in margins.items() if value == margin_bin and (profile == "all" or scenario_by_id[sid].profile == profile)]
                rows = [r for r in candidates if margins[r["scenario_id"]] == margin_bin and (profile == "all" or r["profile"] == profile) and (kind == "all" or r["perturbation_class"] == kind)]
                record: dict[str, Any] = {"threshold_margin_bin": margin_bin, "profile": profile,
                                          "perturbation_class": kind, "baseline_scenario_count": len(scenario_ids)}
                record.update(_aggregate(rows))
                output.append(record)
    return output


def _comparison_rows(results: Sequence[Mapping[str, Any]], vector_set: str | None = None) -> list[Mapping[str, Any]]:
    return [row for row in results if vector_set is None or row["vector_set"] == vector_set]


def _capture_hashes(repo_root: Path, source_dir: Path) -> dict[str, str]:
    paths = [repo_root / item for item in HASHED_INPUTS]
    paths.extend(source_dir / item for item in SOURCE_FILENAMES)
    paths.append(Path(__file__).resolve())
    return {_relative(path, repo_root): sha256_file(path) for path in paths}


def _check_invariants(results: Sequence[Mapping[str, Any]], scenarios: Sequence[Scenario], vectors: Sequence[WeightVector],
                      policy: Mapping[str, Any], before: Mapping[str, str], after: Mapping[str, str],
                      repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    failures = []
    acute = [row for row in results if row["scenario_name"] == "strong_acute_symptoms_learned_low"]
    no_evidence = [row for row in results if row["scenario_name"] == "no_usable_evidence" or
                   (row["scenario_family"] == "availability_mask" and row["available_modality_count"] == 0)]
    for row in acute:
        if row["perturbed_status"] != "completed" or row["perturbed_risk_band"] != "URGENT" or row["perturbed_fused_score"] is None or row["perturbed_fused_score"] < policy["hard_escalation_score_floor"]:
            failures.append({"invariant": "acute_escalation", "scenario_id": row["scenario_id"],
                             "weight_vector_id": row["weight_vector_id"], "status": row["perturbed_status"],
                             "score": row["perturbed_fused_score"], "risk_band": row["perturbed_risk_band"]})
    for row in no_evidence:
        if row["perturbed_status"] != "no_usable_evidence" or row["perturbed_fused_score"] is not None or row["perturbed_risk_band"] is not None:
            failures.append({"invariant": "no_usable_evidence", "scenario_id": row["scenario_id"],
                             "weight_vector_id": row["weight_vector_id"], "status": row["perturbed_status"]})
    for row in results:
        if row["perturbed_status"] == "completed":
            score = row["perturbed_fused_score"]
            if score is None or not math.isfinite(float(score)) or not 0 <= float(score) <= 1:
                failures.append({"invariant": "finite_bounded_score", "scenario_id": row["scenario_id"],
                                 "weight_vector_id": row["weight_vector_id"], "score": score})
        expected = transition_label(row["baseline_risk_band"], row["perturbed_risk_band"])
        if row["transition_label"] != expected:
            failures.append({"invariant": "transition_label", "scenario_id": row["scenario_id"],
                             "weight_vector_id": row["weight_vector_id"], "expected": expected,
                             "actual": row["transition_label"]})
    for vector in vectors:
        try:
            _validate_weights(vector.weights, vector.vector_id)
        except ValueError as exc:
            failures.append({"invariant": "weight_validity", "weight_vector_id": vector.vector_id, "error": str(exc)})
    hashes_same = dict(before) == dict(after)
    if not hashes_same:
        failures.append({"invariant": "hashes_unchanged", "before": dict(before), "after": dict(after)})
    module_weights = dict(DEFAULT_FUSION_WEIGHTS)
    policy_after = load_frozen_policy(repo_root)
    weights_same = module_weights == policy["baseline_weights"] == policy_after["baseline_weights"]
    if not weights_same:
        failures.append({"invariant": "production_weights_unchanged", "module": module_weights,
                         "registry": policy_after["baseline_weights"]})
    errors = [row for row in results if row["perturbed_status"] == "fusion_error"]
    return {
        "all_pass": not failures, "failures": failures,
        "weight_validity": {"simplex_count": sum(v.vector_set == "simplex" for v in vectors),
                             "targeted_count": sum(v.vector_set == "targeted" for v in vectors),
                             "all_nonnegative_sum_one": not any(f["invariant"] == "weight_validity" for f in failures)},
        "acute_escalation": {"scenario_count": len({r["scenario_id"] for r in acute}),
                             "vector_comparisons": len(acute),
                             "invariant_pass": not any(f["invariant"] == "acute_escalation" for f in failures)},
        "no_usable_evidence": {"scenario_count": len({r["scenario_id"] for r in no_evidence}),
                               "vector_comparisons": len(no_evidence),
                               "invariant_pass": not any(f["invariant"] == "no_usable_evidence" for f in failures)},
        "production_immutability": {"hashes_unchanged": hashes_same,
                                    "registry_and_module_weights_unchanged": weights_same,
                                    "hashes_before": dict(before), "hashes_after": dict(after)},
        "score_bounds_and_transition_labels_pass": not any(f["invariant"] in ("finite_bounded_score", "transition_label") for f in failures),
        "recorded_fusion_errors": {"count": len(errors),
                                   "reason_counts": dict(Counter(r["fusion_error_message"] for r in errors)),
                                   "policy_behavior": "Production fusion errors for zero total available candidate weight are recorded without repair."},
        "scenario_count": len(scenarios),
    }


def _determinism_check(scenarios: Sequence[Scenario], vectors: Sequence[WeightVector], policy: Mapping[str, Any]) -> dict[str, Any]:
    pairs = [
        ("conflicting_face_and_speech", "reference", "SW-000"),
        ("missing_face", "edge", "SW-001"),
        ("symptoms_only", "reference", "SW-084"),
        ("conflicting_face_and_speech__mask_0111", "edge", "TW-metadata_context-m10"),
        ("no_usable_evidence", "reference", "SW-000"),
    ]
    scenario_map = {(s.name, s.profile): s for s in scenarios}
    vector_map = {v.vector_id: v for v in vectors}
    checked = []
    for name, profile, vector_id in pairs:
        scenario, vector = scenario_map[(name, profile)], vector_map[vector_id]
        first = evaluate_scenario(scenario, vector.weights, policy)
        second = evaluate_scenario(scenario, vector.weights, policy)
        if any(first[key] != second[key] for key in ("fused_score", "risk_band", "status", "effective_weights")):
            raise AssertionError(f"Repeated execution differed for {scenario.scenario_id}, {vector_id}.")
        checked.append({"scenario_id": scenario.scenario_id, "weight_vector_id": vector_id})
    return {"pass": True, "representative_pairs_repeated": len(checked), "pairs": checked}


def _csv_cell(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return "" if value is None else value


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty output table: {path}.")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        writer.writerows({key: _csv_cell(value) for key, value in row.items()} for row in rows)


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def _plot_profile_stability(summary: Sequence[Mapping[str, Any]], path: Path) -> None:
    profiles, classes = ("edge", "reference"), ("local_5pp", "stress_5_to_10pp")
    x, width = range(2), 0.34
    fig, ax = plt.subplots(figsize=(7.2, 4.4), constrained_layout=True)
    for i, kind in enumerate(classes):
        values = [next(r for r in summary if r["grouping"] == "profile" and r["profile"] == p and r["perturbation_class"] == kind)["band_stability"] for p in profiles]
        ax.bar([v + (i - 0.5) * width for v in x], [float(v) if v is not None else math.nan for v in values], width, label=kind)
    ax.set_xticks(list(x), profiles); ax.set_ylim(0, 1.02); ax.set_ylabel("Band stability"); ax.set_xlabel("Evidence profile")
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(.5, 1.0), ncol=2); ax.grid(axis="y", alpha=.25); ax.set_axisbelow(True)
    path.parent.mkdir(parents=True, exist_ok=True); fig.savefig(path, dpi=180); plt.close(fig)


def _plot_availability_stability(summary: Sequence[Mapping[str, Any]], path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.3), sharey=True, constrained_layout=True)
    for ax, profile in zip(axes, ("edge", "reference")):
        for kind in ("local_5pp", "stress_5_to_10pp"):
            rows = sorted([r for r in summary if r["grouping"] == "profile_family_availability" and r["profile"] == profile and r["scenario_family"] == "availability_mask" and r["perturbation_class"] == kind and r["available_modality_count"] != "all" and r["band_stability"] is not None], key=lambda r: int(r["available_modality_count"]))
            ax.plot([int(r["available_modality_count"]) for r in rows], [r["band_stability"] for r in rows], marker="o", label=kind)
        ax.set_title(profile); ax.set_xlabel("Available modalities"); ax.set_xticks((1, 2, 3, 4)); ax.set_ylim(0, 1.02); ax.grid(alpha=.25)
    axes[0].set_ylabel("Band stability"); axes[1].legend(frameon=False, loc="lower right")
    path.parent.mkdir(parents=True, exist_ok=True); fig.savefig(path, dpi=180); plt.close(fig)


def _plot_margin(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.6), constrained_layout=True); x, width = range(4), .34
    for i, kind in enumerate(("local_5pp", "stress_5_to_10pp")):
        selected = [next(r for r in rows if r["threshold_margin_bin"] == b and r["profile"] == "all" and r["perturbation_class"] == kind) for b in MARGIN_BINS]
        ax.bar([v + (i - .5) * width for v in x], [r["band_flip_rate"] if r["band_flip_rate"] is not None else math.nan for r in selected], width, label=kind)
    ax.set_xticks(list(x), MARGIN_BINS); ax.set_ylabel("Band flip rate"); ax.set_xlabel("Baseline distance to nearest ordinary threshold"); ax.set_ylim(0, 1.02)
    ax.legend(frameon=False); ax.grid(axis="y", alpha=.25); ax.set_axisbelow(True)
    path.parent.mkdir(parents=True, exist_ok=True); fig.savefig(path, dpi=180); plt.close(fig)


def _plot_targeted(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 4.8), constrained_layout=True); x, width = range(4), .18
    for i, delta in enumerate((-.10, -.05, .05, .10)):
        values = [next(r for r in rows if r["modality"] == modality and r["delta"] == delta and r["profile"] == "all")["mean_absolute_score_change"] for modality in WEIGHT_NAMES]
        ax.bar([v + (i - 1.5) * width for v in x], [value if value is not None else math.nan for value in values], width, label=f"{delta:+.2f}")
    ax.set_xticks(list(x), WEIGHT_NAMES); ax.set_ylabel("Mean absolute fused-score change"); ax.set_xlabel("Target modality")
    ax.legend(title="Target weight change", frameon=False, loc="lower center", bbox_to_anchor=(.5, 1.0), ncol=4); ax.grid(axis="y", alpha=.25); ax.set_axisbelow(True)
    path.parent.mkdir(parents=True, exist_ok=True); fig.savefig(path, dpi=180); plt.close(fig)


def _pct(value: Any) -> str:
    return "n/a" if value is None else f"{100 * float(value):.1f}%"


def _num(value: Any, digits: int = 4) -> str:
    return "n/a" if value is None else f"{float(value):.{digits}f}"


def _table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    lines.extend("| " + " | ".join(str(value) for value in row) + " |" for row in rows)
    return "\n".join(lines)


def render_report(policy: Mapping[str, Any], manifest: Mapping[str, Any], scenarios: Sequence[Scenario],
                  vectors: Sequence[WeightVector], summary: Sequence[Mapping[str, Any]],
                  targeted: Sequence[Mapping[str, Any]], margins: Sequence[Mapping[str, Any]],
                  results: Sequence[Mapping[str, Any]], invariants: Mapping[str, Any]) -> str:
    def metric(grouping: str, kind: str, profile: str = "all", family: str = "all") -> Mapping[str, Any]:
        return next(r for r in summary if r["grouping"] == grouping and r["perturbation_class"] == kind and r["profile"] == profile and r["scenario_family"] == family)
    local, stress = metric("overall", "local_5pp"), metric("overall", "stress_5_to_10pp")
    profile_table = []
    for profile in ("edge", "reference"):
        l, s = metric("profile", "local_5pp", profile), metric("profile", "stress_5_to_10pp", profile)
        profile_table.append((profile, _pct(l["band_stability"]), _pct(s["band_stability"]), _num(s["mean_absolute_score_change"])))
    availability_table = []
    for count in range(5):
        values = []
        for profile in ("edge", "reference"):
            for kind in ("local_5pp", "stress_5_to_10pp"):
                row = next(r for r in summary if r["grouping"] == "profile_family_availability" and r["profile"] == profile and r["scenario_family"] == "availability_mask" and r["available_modality_count"] == count and r["perturbation_class"] == kind)
                values.append(
                    f"{profile} {kind}: {_pct(row['band_stability'])} "
                    f"({row['completed_band_comparisons']}/{row['perturbation_comparisons']} completed; "
                    f"{row['fusion_error_count']} fusion errors; "
                    f"{row['no_usable_evidence_count']} no-usable-evidence)"
                )
        availability_table.append((count, "<br>".join(values)))
    two_modalities = [r for r in summary if r["grouping"] == "profile_family_availability" and r["scenario_family"] == "availability_mask" and r["available_modality_count"] == 2 and r["band_stability"] is not None]
    four_modalities = [r for r in summary if r["grouping"] == "profile_family_availability" and r["scenario_family"] == "availability_mask" and r["available_modality_count"] == 4 and r["band_stability"] is not None]
    two_modality_range = (min(r["band_stability"] for r in two_modalities), max(r["band_stability"] for r in two_modalities))
    four_modality_range = (min(r["band_stability"] for r in four_modalities), max(r["band_stability"] for r in four_modalities))
    target_table = []
    for modality in WEIGHT_NAMES:
        for delta in (-.10, -.05, .05, .10):
            row = next(r for r in targeted if r["modality"] == modality and r["delta"] == delta and r["profile"] == "all")
            transitions = row["transition_counts_json"]
            text = ", ".join(f"{key}: {value}" for key, value in transitions.items()) if transitions else "none"
            target_table.append((modality, f"{delta:+.2f}", _pct(row["band_stability"]), _pct(row["band_flip_rate"]), f"{row['completed_band_comparisons']}/{row['perturbation_comparisons']} completed; {row['fusion_error_count']} fusion errors; {row['no_usable_evidence_count']} no-usable-evidence", _num(row["mean_absolute_score_change"]), _num(row["maximum_absolute_score_change"]), text))
    margin_table = []
    for bin_label in MARGIN_BINS:
        l = next(r for r in margins if r["threshold_margin_bin"] == bin_label and r["profile"] == "all" and r["perturbation_class"] == "local_5pp")
        s = next(r for r in margins if r["threshold_margin_bin"] == bin_label and r["profile"] == "all" and r["perturbation_class"] == "stress_5_to_10pp")
        margin_table.append((bin_label, l["baseline_scenario_count"], f"{l['completed_band_comparisons']}/{l['perturbation_comparisons']} completed; {l['fusion_error_count']} errors", _pct(l["band_flip_rate"]), _num(l["mean_absolute_score_change"]), f"{s['completed_band_comparisons']}/{s['perturbation_comparisons']} completed; {s['fusion_error_count']} errors", _pct(s["band_flip_rate"]), _num(s["mean_absolute_score_change"])))
    local_margin_rows = [r for r in margins if r["profile"] == "all" and r["perturbation_class"] == "local_5pp" and r["band_flip_rate"] is not None]
    stress_margin_rows = [r for r in margins if r["profile"] == "all" and r["perturbation_class"] == "stress_5_to_10pp" and r["band_flip_rate"] is not None]
    local_peak = max(local_margin_rows, key=lambda r: r["band_flip_rate"])
    stress_peak = max(stress_margin_rows, key=lambda r: r["band_flip_rate"])
    local_nearest = next(r for r in local_margin_rows if r["threshold_margin_bin"] == MARGIN_BINS[0])
    stress_nearest = next(r for r in stress_margin_rows if r["threshold_margin_bin"] == MARGIN_BINS[0])
    scenario_metrics = build_scenario_sensitivity(results)
    top = sorted((r for r in scenario_metrics if r["maximum_absolute_score_change"] is not None),
                 key=lambda r: (-r["maximum_absolute_score_change"], -(r["mean_absolute_score_change"] or 0), r["profile"], r["scenario_name"]))[:5]
    top_rows = [(r["profile"], r["scenario_family"], r["scenario_name"], r["perturbation_class"], _num(r["mean_absolute_score_change"]), _num(r["maximum_absolute_score_change"])) for r in top]
    transition_summary = build_band_transitions(results)
    transition_table = [
        (kind, row["transition_label"], row["count"])
        for kind in ("local_5pp", "stress_5_to_10pp")
        for row in transition_summary
        if row["comparison_set"] == "simplex" and row["profile"] == "all"
        and row["scenario_family"] == "all" and row["perturbation_class"] == kind
        and int(row["count"]) > 0
    ]
    targeted_groups = [r for r in targeted if r["profile"] == "all" and r["mean_absolute_score_change"] is not None]
    most_sensitive_target = max(targeted_groups, key=lambda r: r["mean_absolute_score_change"])
    max_shift = max((r["absolute_score_delta"] for r in results if r["absolute_score_delta"] is not None), default=0.0)
    family_counts = Counter(s.scenario_family for s in scenarios)
    vector_counts = Counter(v.perturbation_class for v in vectors if v.vector_set == "simplex")
    hash_table = [f"| {path} | {digest} |" for path, digest in manifest["source_hashes"].items()]
    errors = invariants["recorded_fusion_errors"]
    return "\n".join([
        "# Deterministic Fusion-Weight Sensitivity Analysis", "",
        "## 1. Objective", "",
        "This experiment measures the deterministic robustness of the current manually specified late-fusion policy under predefined modality-weight changes. It does not select, optimize, or recommend a weight vector.", "",
        "## 2. Frozen policy", "",
        f"The baseline registry and canonical implementation agree on face={policy['baseline_weights']['face']:.2f}, speech={policy['baseline_weights']['speech']:.2f}, acute symptoms={policy['baseline_weights']['acute_symptoms']:.2f}, and metadata/context={policy['baseline_weights']['metadata_context']:.2f}; the weights sum to 1.00. Ordinary thresholds are 0.50 and 0.75. Available modality weights are renormalized over available scores. Hard acute escalation retains the 0.85 score floor and URGENT band. No usable evidence retains the insufficient-evidence status.", "",
        "Weights remain manually specified. This experiment measures deterministic policy robustness only. Its scenarios are synthetic/representative or validation-grounded, not paired clinical patients. No patient-level multimodal clinical ground truth exists, so this analysis cannot validate or optimize weights clinically.", "",
        "## 3. Perturbation methodology", "",
        f"The 0.05-grid local simplex envelope contains 19 vectors including baseline, with {vector_counts['local_5pp']} non-baseline local perturbations. The extended envelope contains 85 vectors total, including {vector_counts['stress_5_to_10pp']} stress perturbations from greater than 5pp through 10pp. The separate targeted subset contains {sum(v.vector_set == 'targeted' for v in vectors)} modality/delta vectors and proportionally redistributes compensating changes across the other baseline weights.", "",
        "Nonempty cases call the canonical fuse_multimodal_scores function with copied candidate weights. Zero-evidence cases retain no_usable_evidence with no score or band. When metadata alone is available and the candidate metadata weight is zero, the canonical function raises because the available weight sum is zero; that result is recorded as a fusion error without repair. Targeted results do not enter the simplex stability denominator.", "",
        "## 4. Scenario sources", "",
        f"Compatibility profile identities are mapped explicitly as canonical → edge and pretrained → reference. Family counts across profiles are hand-authored={family_counts['hand_authored']}, distribution-grounded={family_counts['distribution_grounded']}, and availability masks={family_counts['availability_mask']}.", "",
        "The 12 requested hand-authored scenarios per profile reuse fusion_scenario_comparison.csv. The 16 stored distribution-grounded combinations per profile reuse distribution_grounded_scenarios.csv without recalculating quantiles. Provenance records no held-out test rows/results used; no model inference or test-prediction artifacts were loaded.", "",
        "| Source artifact | SHA-256 |", "|---|---|", *hash_table, "",
        "## 5. Local ±5pp results", "",
        f"Across the 18 non-baseline local vectors, band stability was {_pct(local['band_stability'])} ({local['completed_band_comparisons']} completed comparisons). Mean, median, and maximum absolute fused-score changes were {_num(local['mean_absolute_score_change'])}, {_num(local['median_absolute_score_change'])}, and {_num(local['maximum_absolute_score_change'])}.", "",
        "## 6. 5–10pp stress results", "",
        f"Across the 66 stress vectors, band stability was {_pct(stress['band_stability'])} ({stress['completed_band_comparisons']} completed comparisons). Mean, median, and maximum absolute fused-score changes were {_num(stress['mean_absolute_score_change'])}, {_num(stress['median_absolute_score_change'])}, and {_num(stress['maximum_absolute_score_change'])}.", "",
        "## 7. Edge versus reference profile", "",
        _table(("Profile", "Local band stability", "Stress band stability", "Stress mean absolute score change"), profile_table), "",
        "These profile differences describe how the same policy perturbations act on the stored evidence-score distributions; they are not comparative clinical performance.", "",
        "## 8. Missing-modality analysis", "",
        _table(("Available modalities", "Band stability and completed/error counts by profile and group"), availability_table), "",
        f"Two-modality masks had band stability from {_pct(two_modality_range[0])} to {_pct(two_modality_range[1])}, compared with {_pct(four_modality_range[0])} to {_pct(four_modality_range[1])} for the four-modality masks. This indicates greater observed band sensitivity under two-modality availability for this fixed scenario tuple. Single-modality stability is conditional on completed cases; zero available weight can prevent fusion.", "",
        f"The complete 16-mask analysis derives from each profile's exact conflicting_face_and_speech tuple and changes availability only. There were {errors['count']} perturbation comparisons where production fusion raised due to zero total available candidate weight. They have no completed band or score and are excluded from completed-band and score-change denominators.", "",
        "## 9. Targeted modality sensitivity", "",
        _table(("Modality", "Delta", "Band stability", "Flip rate", "Completed comparisons and errors", "Mean absolute score change", "Maximum absolute score change", "Transitions"), target_table), "",
        f"The largest targeted mean absolute score change was observed for {most_sensitive_target['modality']} at delta {most_sensitive_target['delta']:+.2f} ({_num(most_sensitive_target['mean_absolute_score_change'])}). This describes policy sensitivity, not feature, predictive, or clinical importance.", "",
        "### Observed simplex band transitions", "",
        _table(("Perturbation group", "Transition", "Count"), transition_table), "",
        "The CSV transition matrix also retains zero-count transitions explicitly.", "",
        "## 10. Threshold-margin findings", "",
        "Margin is the distance from the baseline fused score to its nearest ordinary threshold. Hard-escalation cases are excluded. Counts, flip rates, and absolute score changes are shown for local and stress comparisons.", "",
        _table(("Baseline margin bin", "Baseline scenarios", "Local completed/total (errors)", "Local flip rate", "Local mean |Δscore|", "Stress completed/total (errors)", "Stress flip rate", "Stress mean |Δscore|"), margin_table), "",
        f"Threshold flips peaked in {local_peak['threshold_margin_bin']} for local perturbations ({_pct(local_peak['band_flip_rate'])}) and {stress_peak['threshold_margin_bin']} for stress perturbations ({_pct(stress_peak['band_flip_rate'])}); in the nearest bin they were {_pct(local_nearest['band_flip_rate'])} and {_pct(stress_nearest['band_flip_rate'])}, respectively.", "",
        "Whether changes concentrate near decision boundaries should be read from these observed counts and rates; no such result is assumed in advance.", "",
        "## 11. Safety invariants", "",
        f"Acute hard escalation: {'PASS' if invariants['acute_escalation']['invariant_pass'] else 'FAIL'} ({invariants['acute_escalation']['vector_comparisons']} comparisons). No usable evidence: {'PASS' if invariants['no_usable_evidence']['invariant_pass'] else 'FAIL'} ({invariants['no_usable_evidence']['vector_comparisons']} comparisons). Deterministic repeat execution: {'PASS' if invariants['determinism']['pass'] else 'FAIL'} ({invariants['determinism']['representative_pairs_repeated']} pairs). Production and source hashes unchanged: {'PASS' if invariants['production_immutability']['hashes_unchanged'] else 'FAIL'}.", "",
        "No risk-band transition direction is labeled favorable or unfavorable.", "",
        "## 12. Limitations", "",
        "The scenarios are synthetic/representative or validation-derived quantile combinations, not paired patient-level multimodal observations. Modality scores have different semantics and are not jointly calibrated clinical probabilities. No paired patient-level multimodal clinical ground truth exists. This analysis cannot validate clinical utility or optimize weights. It does not compute diagnostic accuracy, sensitivity, specificity, ROC-AUC, or PR-AUC for fused outcomes.", "",
        "## 13. Concise report-ready findings", "",
        f"- Local ±5pp band stability was {_pct(local['band_stability'])}; stress 5–10pp stability was {_pct(stress['band_stability'])}.",
        f"- Maximum absolute fused-score change across completed simplex and targeted comparisons was {max_shift:.4f}.",
        "- Largest scenario-level maximum shifts were observed in these descriptive rows:",
        *[f"  - {profile}, {family}, {name}, {kind}: mean {_num(mean)}, maximum {_num(maximum)}." for profile, family, name, kind, mean, maximum in top_rows],
        f"- Among targeted modality/delta groups, {most_sensitive_target['modality']} at {most_sensitive_target['delta']:+.2f} had the largest mean absolute score change ({_num(most_sensitive_target['mean_absolute_score_change'])}).",
        "- Threshold-margin findings are presented by bin and should be interpreted from the counts and flip rates above.",
        f"- Acute escalation and no-usable-evidence invariants were {'preserved' if invariants['all_pass'] else 'not fully preserved'}; weights remain manual and clinical validation or optimization is not supported without paired multimodal ground truth.", "",
        "## Figures", "", "- stability_by_profile.png", "- stability_by_availability.png", "- threshold_margin_sensitivity.png", "- targeted_modality_sensitivity.png", "",
    ])


def run_analysis(repo_root: Path = REPO_ROOT, source_dir: Path = DEFAULT_SOURCE_DIR,
                 output_dir: Path = DEFAULT_OUTPUT_DIR, report_path: Path = DEFAULT_REPORT_PATH) -> dict[str, Any]:
    repo_root, source_dir, output_dir, report_path = (p.resolve() for p in (repo_root, source_dir, output_dir, report_path))
    policy = load_frozen_policy(repo_root)
    source = load_compatibility_sources(source_dir)
    scenarios = build_all_scenarios(source)
    simplex, targeted = generate_simplex_vectors(.10), generate_targeted_vectors()
    vectors = [*simplex, *targeted]
    before = _capture_hashes(repo_root, source_dir)
    results, baselines = build_sensitivity_results(scenarios, simplex, targeted, policy)
    deterministic = _determinism_check(scenarios, vectors, policy)
    after = _capture_hashes(repo_root, source_dir)
    invariants = _check_invariants(results, scenarios, vectors, policy, before, after, repo_root)
    invariants["determinism"] = deterministic
    if not deterministic["pass"]:
        invariants["all_pass"] = False
        invariants["failures"].append({"invariant": "determinism", "details": deterministic})

    summary = build_stability_summary(results)
    target_summary = build_targeted_modality_summary(results)
    transitions = build_band_transitions(results)
    scenario_summary = build_scenario_sensitivity(results)
    margin_summary = build_threshold_margin_analysis(results, baselines, scenarios, policy)
    figures = output_dir / "figures"
    _plot_profile_stability(summary, figures / "stability_by_profile.png")
    _plot_availability_stability(summary, figures / "stability_by_availability.png")
    _plot_margin(margin_summary, figures / "threshold_margin_sensitivity.png")
    _plot_targeted(target_summary, figures / "targeted_modality_sensitivity.png")

    hashes_source = {_relative(path, repo_root): sha256_file(path) for path in source["loaded_paths"]}
    family_counts = Counter(s.scenario_family for s in scenarios)
    profile_counts = {profile: dict(Counter(s.scenario_family for s in scenarios if s.profile == profile)) for profile in PROFILE_MAPPING.values()}
    vector_counts = Counter(v.perturbation_class for v in simplex)
    config = {
        "objective": "Deterministic sensitivity analysis of the current manual fusion policy; not optimization.",
        "baseline_weights": policy["baseline_weights"], "weight_order": list(WEIGHT_NAMES),
        "simplex_increment": .05, "local_max_absolute_delta": .05, "extended_max_absolute_delta": .10,
        "simplex_vector_counts": {"total": len(simplex), **dict(vector_counts)},
        "targeted_vector_count": len(targeted), "targeted_deltas": [-.10, -.05, .05, .10],
        "profile_mapping": PROFILE_MAPPING, "scenario_family_counts": dict(family_counts),
        "scenario_family_counts_by_profile": profile_counts,
        "thresholds": {"moderate": policy["moderate_threshold"], "high": policy["high_threshold"]},
        "hard_escalation_score_floor": policy["hard_escalation_score_floor"],
        "source_directory": _relative(source_dir, repo_root), "held_out_test_prediction_artifacts_loaded": False,
        "paired_patient_level_multimodal_ground_truth_available": False,
        "primary_denominator": "Simplex local/stress comparisons only; baseline and targeted vectors excluded.",
    }
    manifest = {
        "audit_id": source["provenance"].get("audit_id"), "created_utc": source["provenance"].get("created_utc"),
        "loaded_files": [_relative(path, repo_root) for path in source["loaded_paths"]],
        "source_hashes": hashes_source, "profile_mapping": PROFILE_MAPPING,
        "selected_models": source["provenance"].get("selected_models", {}),
        "source_artifacts_from_provenance": source["provenance"].get("source_artifacts", []),
        "test_rows_used": source["provenance"].get("test_rows_used"),
        "test_results_used_for_tuning": source["provenance"].get("test_results_used_for_tuning"),
        "held_out_test_prediction_artifacts_loaded": False,
        "note": "Execution opens only the two scenario CSVs and the compatibility provenance JSON.",
    }
    outputs: dict[str, Any] = {
        "analysis_config.json": config, "source_manifest.json": manifest,
        "weight_vectors.csv": [_vector_record(v) for v in vectors],
        "scenario_manifest.csv": [_scenario_record(s) for s in scenarios],
        "sensitivity_results.csv": results, "stability_summary.csv": summary,
        "targeted_modality_sensitivity.csv": target_summary, "band_transitions.csv": transitions,
        "scenario_sensitivity.csv": scenario_summary, "threshold_margin_analysis.csv": margin_summary,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    for filename, rows in outputs.items():
        path = output_dir / filename
        if filename.endswith(".json"):
            _write_json(path, rows)
        else:
            _write_csv(path, rows)
    _write_json(output_dir / "invariants.json", invariants)
    _write_json(output_dir / "artifact_hashes.json", {"sha256_before": before, "sha256_after": after, "all_unchanged": before == after})
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(render_report(policy, manifest, scenarios, vectors, summary, target_summary,
                                         margin_summary, results, invariants), encoding="utf-8")
    result = {"output_dir": output_dir, "report_path": report_path, "scenario_count": len(scenarios),
              "scenario_family_counts": dict(family_counts), "profile_counts": profile_counts,
              "simplex_vector_count": len(simplex), "simplex_class_counts": dict(vector_counts),
              "targeted_vector_count": len(targeted), "result_row_count": len(results), "invariants": invariants,
              "hashes_before": before, "hashes_after": after}
    if not invariants["all_pass"]:
        raise AssertionError("Sensitivity invariant failure; see invariants.json for exact scenarios/vectors.")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run deterministic fusion-weight sensitivity analysis.")
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT_PATH)
    args = parser.parse_args(argv)
    try:
        result = run_analysis(args.repo_root, args.source_dir, args.output_dir, args.report_path)
    except (AssertionError, FileNotFoundError, PolicyMismatchError, ValueError) as exc:
        print(f"Fusion-weight sensitivity analysis failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({"output_dir": str(result["output_dir"]), "report_path": str(result["report_path"]),
                      "scenario_count": result["scenario_count"], "scenario_family_counts": result["scenario_family_counts"],
                      "profile_counts": result["profile_counts"], "simplex_vector_count": result["simplex_vector_count"],
                      "simplex_class_counts": result["simplex_class_counts"], "targeted_vector_count": result["targeted_vector_count"],
                      "result_row_count": result["result_row_count"], "invariants_pass": result["invariants"]["all_pass"],
                      "fusion_error_count": result["invariants"]["recorded_fusion_errors"]["count"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
