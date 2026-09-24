from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from scripts import fusion_weight_sensitivity_analysis as analysis


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = (
    REPO_ROOT
    / "reports"
    / "experiments"
    / "pretrained_reference_profile_compatibility_audit"
)
EXPECTED_HAND_SCENARIOS = {
    "all_evidence_low",
    "face_high_others_low",
    "speech_high_others_low",
    "metadata_high_acute_low",
    "face_speech_corroborating_high",
    "strong_acute_symptoms_learned_low",
    "conflicting_face_and_speech",
    "missing_face",
    "missing_speech",
    "missing_metadata",
    "symptoms_only",
    "no_usable_evidence",
}


def test_policy_uses_exact_frozen_baseline_and_thresholds() -> None:
    policy = analysis.load_frozen_policy(REPO_ROOT)

    assert policy["baseline_weights"] == {
        "face": 0.35,
        "speech": 0.30,
        "acute_symptoms": 0.25,
        "metadata_context": 0.10,
    }
    assert sum(policy["baseline_weights"].values()) == pytest.approx(1.0, abs=1e-12)
    assert policy["moderate_threshold"] == 0.50
    assert policy["high_threshold"] == 0.75
    assert policy["hard_escalation_score_floor"] == 0.85


@pytest.mark.parametrize(
    ("envelope", "expected_count"),
    [(0.05, 19), (0.10, 85)],
)
def test_simplex_sweep_has_exact_grid_count_and_valid_weights(
    envelope: float, expected_count: int
) -> None:
    vectors = analysis.generate_simplex_vectors(envelope)

    assert len(vectors) == expected_count
    assert sum(vector.perturbation_class == "baseline" for vector in vectors) == 1
    for vector in vectors:
        assert set(vector.weights) == set(analysis.WEIGHT_NAMES)
        assert all(weight >= 0 for weight in vector.weights.values())
        assert sum(vector.weights.values()) == pytest.approx(1.0, abs=1e-12)
        assert all(
            round(weight / 0.05) * 0.05 == pytest.approx(weight, abs=1e-12)
            for weight in vector.weights.values()
        )


def test_targeted_vectors_preserve_non_target_ratios_and_allow_zero_metadata() -> None:
    vectors = analysis.generate_targeted_vectors()

    assert len(vectors) == 16
    baseline = analysis.BASELINE_WEIGHTS
    for vector in vectors:
        target = vector.target_modality
        assert target is not None
        non_targets = [name for name in analysis.WEIGHT_NAMES if name != target]
        ratios = [vector.weights[name] / baseline[name] for name in non_targets]
        assert ratios == pytest.approx([ratios[0]] * 3, abs=1e-12)
        assert all(weight >= 0 for weight in vector.weights.values())
        assert sum(vector.weights.values()) == pytest.approx(1.0, abs=1e-12)
    metadata_minus_tenth = next(
        item
        for item in vectors
        if item.target_modality == "metadata_context" and item.target_delta == -0.10
    )
    assert metadata_minus_tenth.weights["metadata_context"] == 0.0


def test_compatibility_scenarios_are_reused_with_explicit_profile_mapping() -> None:
    source = analysis.load_compatibility_sources(SOURCE_DIR)
    hand = source["hand_authored"]
    quantile = source["distribution_grounded"]

    assert len(hand) == 24
    assert {scenario.name for scenario in hand} == EXPECTED_HAND_SCENARIOS
    assert {scenario.profile for scenario in hand} == {"edge", "reference"}
    assert len(quantile) == 32
    assert len({scenario.name for scenario in quantile if scenario.profile == "edge"}) == 16
    assert len({scenario.name for scenario in quantile if scenario.profile == "reference"}) == 16
    assert source["profile_mapping"] == {"canonical": "edge", "pretrained": "reference"}
    reference_low = next(
        item for item in hand
        if item.name == "all_evidence_low" and item.profile == "reference"
    )
    assert reference_low.scores["speech"] == pytest.approx(2.0744423826726132e-10)


def test_source_loader_opens_only_scenarios_and_provenance(monkeypatch: pytest.MonkeyPatch) -> None:
    opened: list[str] = []
    real_open = Path.open

    def tracking_open(path: Path, *args: object, **kwargs: object):
        opened.append(path.name)
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", tracking_open)
    analysis.load_compatibility_sources(SOURCE_DIR)

    assert set(opened) == {
        "fusion_scenario_comparison.csv",
        "distribution_grounded_scenarios.csv",
        "provenance_manifest.json",
    }
    assert len(opened) == 3


def test_scenario_set_adds_all_sixteen_masks_from_each_conflict_tuple() -> None:
    source = analysis.load_compatibility_sources(SOURCE_DIR)
    scenarios = analysis.build_all_scenarios(source)
    masks = [item for item in scenarios if item.scenario_family == "availability_mask"]

    assert len(masks) == 32
    for profile in ("edge", "reference"):
        current = [item for item in masks if item.profile == profile]
        assert len(current) == 16
        assert sorted(item.available_modality_count for item in current) == [
            0,
            *([1] * 4),
            *([2] * 6),
            *([3] * 4),
            4,
        ]
        full = next(item for item in current if item.available_modality_count == 4)
        conflict = next(
            item for item in source["hand_authored"]
            if item.name == "conflicting_face_and_speech" and item.profile == profile
        )
        assert full.scores == conflict.scores
        for mask_case in current:
            for modality, value in mask_case.scores.items():
                if value is not None:
                    assert value == conflict.scores[modality]


def test_evaluation_calls_the_canonical_fusion_function(monkeypatch: pytest.MonkeyPatch) -> None:
    scenario = next(
        item
        for item in analysis.load_compatibility_sources(SOURCE_DIR)["hand_authored"]
        if item.name == "all_evidence_low" and item.profile == "edge"
    )
    policy = analysis.load_frozen_policy(REPO_ROOT)
    calls: list[dict[str, float]] = []
    production_fusion = analysis.fuse_multimodal_scores

    def tracked_fusion(fusion_input, weights=None, **kwargs):
        calls.append(dict(weights or {}))
        return production_fusion(fusion_input, weights=weights, **kwargs)

    monkeypatch.setattr(analysis, "fuse_multimodal_scores", tracked_fusion)
    result = analysis.evaluate_scenario(scenario, policy["baseline_weights"], policy)

    assert len(calls) == 1
    assert calls[0] == policy["baseline_weights"]
    assert result["status"] == "completed"
    assert result["risk_band"] == "LOW"
    assert 0.0 <= result["fused_score"] <= 1.0


def test_acute_and_no_evidence_invariants_hold_for_every_valid_vector() -> None:
    sources = analysis.load_compatibility_sources(SOURCE_DIR)
    scenarios = analysis.build_all_scenarios(sources)
    policy = analysis.load_frozen_policy(REPO_ROOT)
    vectors = [
        *analysis.generate_simplex_vectors(0.10),
        *analysis.generate_targeted_vectors(),
    ]
    acute_cases = [
        item for item in scenarios if item.name == "strong_acute_symptoms_learned_low"
    ]
    no_evidence_cases = [item for item in scenarios if item.name == "no_usable_evidence"]
    zero_mask_cases = [
        item
        for item in scenarios
        if item.scenario_family == "availability_mask"
        and item.available_modality_count == 0
    ]

    for vector in vectors:
        for scenario in acute_cases:
            result = analysis.evaluate_scenario(scenario, vector.weights, policy)
            assert result["status"] == "completed"
            assert result["risk_band"] == "URGENT"
            assert result["fused_score"] >= 0.85
        for scenario in [*no_evidence_cases, *zero_mask_cases]:
            result = analysis.evaluate_scenario(scenario, vector.weights, policy)
            assert result["status"] == "no_usable_evidence"
            assert result["risk_band"] is None
            assert result["fused_score"] is None


def test_representative_replay_is_deterministic_and_hashes_stay_fixed() -> None:
    policy = analysis.load_frozen_policy(REPO_ROOT)
    sources = analysis.load_compatibility_sources(SOURCE_DIR)
    scenarios = analysis.build_all_scenarios(sources)
    vectors = [analysis.generate_simplex_vectors(0.10)[0], *analysis.generate_targeted_vectors()[-1:]]
    chosen = [
        next(item for item in scenarios if item.name == "conflicting_face_and_speech" and item.profile == "reference"),
        next(item for item in scenarios if item.name == "missing_face" and item.profile == "edge"),
    ]
    paths = [
        REPO_ROOT / "rural_stroke_assist/modules/fusion_module.py",
        REPO_ROOT / "config/baseline_registry.json",
        SOURCE_DIR / "fusion_scenario_comparison.csv",
        SOURCE_DIR / "distribution_grounded_scenarios.csv",
    ]
    before = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    for scenario in chosen:
        for vector in vectors:
            first = analysis.evaluate_scenario(scenario, vector.weights, policy)
            second = analysis.evaluate_scenario(scenario, vector.weights, policy)
            for field in ("fused_score", "risk_band", "status", "effective_weights"):
                assert first[field] == second[field]
    after = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    assert before == after


def test_transition_labels_and_threshold_margin_boundaries() -> None:
    assert analysis.transition_label("LOW", "MODERATE") == "LOW->MODERATE"
    assert analysis.transition_label("MODERATE", "LOW") == "MODERATE->LOW"
    assert analysis.transition_label(None, None) is None
    assert analysis.threshold_margin_bin(0.0) == "[0.000, 0.025)"
    assert analysis.threshold_margin_bin(0.025) == "[0.025, 0.050)"
    assert analysis.threshold_margin_bin(0.050) == "[0.050, 0.100)"
    assert analysis.threshold_margin_bin(0.100) == "[0.100, infinity)"


def test_replayed_comparisons_have_bounded_scores_and_actual_transition_labels() -> None:
    sources = analysis.load_compatibility_sources(SOURCE_DIR)
    scenario = next(
        item
        for item in sources["hand_authored"]
        if item.name == "conflicting_face_and_speech" and item.profile == "reference"
    )
    simplex = analysis.generate_simplex_vectors(0.10)
    rows, _ = analysis.build_sensitivity_results(
        [scenario], simplex, [], analysis.load_frozen_policy(REPO_ROOT)
    )

    assert len(rows) == 85
    for row in rows:
        score = row["perturbed_fused_score"]
        assert score is not None and 0.0 <= score <= 1.0
        assert row["transition_label"] == (
            f"{row['baseline_risk_band']}->{row['perturbed_risk_band']}"
        )


def test_stability_summary_can_isolate_full_mask_sensitivity_and_excludes_targeted_rows() -> None:
    def row(vector_set: str, changed: bool, delta: float) -> dict[str, object]:
        return {
            "vector_set": vector_set,
            "perturbation_class": "local_5pp" if vector_set == "simplex" else "targeted_modality",
            "profile": "edge",
            "scenario_family": "availability_mask",
            "available_modality_count": 2,
            "scenario_id": "availability_mask|edge|mask_0011",
            "baseline_status": "completed",
            "perturbed_status": "completed",
            "baseline_risk_band": "LOW",
            "perturbed_risk_band": "MODERATE" if changed else "LOW",
            "band_changed": changed,
            "absolute_score_delta": delta,
        }

    summary = analysis.build_stability_summary(
        [row("simplex", False, 0.01), row("targeted", True, 0.90)]
    )
    isolated = next(
        item
        for item in summary
        if item["grouping"] == "profile_family_availability"
        and item["profile"] == "edge"
        and item["scenario_family"] == "availability_mask"
        and item["available_modality_count"] == 2
    )

    assert isolated["perturbation_class"] == "local_5pp"
    assert isolated["perturbation_comparisons"] == 1
    assert isolated["band_stability"] == 1.0
    assert isolated["mean_absolute_score_change"] == 0.01


def test_production_fusion_inputs_remain_unchanged_after_sample_calls() -> None:
    policy_before = analysis.load_frozen_policy(REPO_ROOT)
    registry_path = REPO_ROOT / "config/baseline_registry.json"
    fusion_path = REPO_ROOT / "rural_stroke_assist/modules/fusion_module.py"
    registry_hash_before = analysis.sha256_file(registry_path)
    fusion_hash_before = analysis.sha256_file(fusion_path)
    scenario = next(
        item
        for item in analysis.load_compatibility_sources(SOURCE_DIR)["hand_authored"]
        if item.name == "face_speech_corroborating_high" and item.profile == "edge"
    )
    analysis.evaluate_scenario(scenario, analysis.generate_simplex_vectors(0.10)[-1].weights, policy_before)

    policy_after = analysis.load_frozen_policy(REPO_ROOT)
    assert policy_after["baseline_weights"] == policy_before["baseline_weights"]
    assert analysis.sha256_file(registry_path) == registry_hash_before
    assert analysis.sha256_file(fusion_path) == fusion_hash_before


def test_human_report_renders_descriptive_scenario_rows() -> None:
    policy = analysis.load_frozen_policy(REPO_ROOT)
    sources = analysis.load_compatibility_sources(SOURCE_DIR)
    scenarios = analysis.build_all_scenarios(sources)
    vectors = [
        *analysis.generate_simplex_vectors(0.10),
        *analysis.generate_targeted_vectors(),
    ]
    summary = []
    for profile in ("all", "edge", "reference"):
        grouping = "overall" if profile == "all" else "profile"
        for kind in ("local_5pp", "stress_5_to_10pp"):
            summary.append({
                "grouping": grouping,
                "profile": profile,
                "scenario_family": "all",
                "perturbation_class": kind,
                "band_stability": 0.9,
                "completed_band_comparisons": 100,
                "mean_absolute_score_change": 0.01,
                "median_absolute_score_change": 0.01,
                "maximum_absolute_score_change": 0.05,
            })
    for profile in ("edge", "reference"):
        for count in range(5):
            for kind in ("local_5pp", "stress_5_to_10pp"):
                comparison_count = (
                    18 if count == 0 and kind == "local_5pp"
                    else 66 if count == 0
                    else 264 if count == 1 and kind == "stress_5_to_10pp"
                    else 72
                )
                completed_count = (
                    0 if count == 0
                    else 249 if count == 1 and kind == "stress_5_to_10pp"
                    else comparison_count
                )
                summary.append({
                    "grouping": "profile_family_availability",
                    "profile": profile,
                    "scenario_family": "availability_mask",
                    "available_modality_count": count,
                    "perturbation_class": kind,
                    "band_stability": 0.8 if count == 2 else 1.0 if count == 4 else 0.9,
                    "perturbation_comparisons": comparison_count,
                    "completed_band_comparisons": completed_count,
                    "fusion_error_count": 15 if count == 1 and kind == "stress_5_to_10pp" else 0,
                    "no_usable_evidence_count": comparison_count if count == 0 else 0,
                })
    targeted = [
        {
            "modality": modality,
            "delta": delta,
            "profile": "all",
            "band_stability": 0.9,
            "band_flip_rate": 0.1,
            "mean_absolute_score_change": 0.01,
            "maximum_absolute_score_change": 0.05,
            "transition_counts_json": {"LOW->MODERATE": 1},
            "completed_band_comparisons": 84 if not (modality == "metadata_context" and delta == -0.10) else 82,
            "perturbation_comparisons": 88,
            "fusion_error_count": 2 if modality == "metadata_context" and delta == -0.10 else 0,
            "no_usable_evidence_count": 4,
        }
        for modality in analysis.WEIGHT_NAMES
        for delta in (-0.10, -0.05, 0.05, 0.10)
    ]
    margins = [
        {
            "threshold_margin_bin": margin_bin,
            "profile": "all",
            "perturbation_class": kind,
            "baseline_scenario_count": 1,
            "perturbation_comparisons": 2,
            "completed_band_comparisons": 2,
                "fusion_error_count": 0,
                "no_usable_evidence_count": 0,
            "band_flip_rate": 0.1,
            "mean_absolute_score_change": 0.01,
        }
        for margin_bin in analysis.MARGIN_BINS
        for kind in ("local_5pp", "stress_5_to_10pp")
    ]
    results = [{
        "scenario_id": "hand_authored|edge|all_evidence_low",
        "scenario_name": "all_evidence_low",
        "scenario_family": "hand_authored",
        "profile": "edge",
        "vector_set": "simplex",
        "perturbation_class": "local_5pp",
        "baseline_status": "completed",
        "perturbed_status": "completed",
        "baseline_risk_band": "LOW",
        "perturbed_risk_band": "LOW",
        "band_changed": False,
        "transition_label": "LOW->LOW",
        "absolute_score_delta": 0.01,
        "available_modality_count": 4,
        "baseline_fused_score": 0.1,
        "perturbed_fused_score": 0.11,
    }]
    invariants = {
        "all_pass": True,
        "acute_escalation": {"invariant_pass": True, "vector_comparisons": 202},
        "no_usable_evidence": {"invariant_pass": True, "vector_comparisons": 404},
        "determinism": {"pass": True, "representative_pairs_repeated": 5},
        "production_immutability": {"hashes_unchanged": True},
        "recorded_fusion_errors": {"count": 0},
    }

    report = analysis.render_report(
        policy,
        {"source_hashes": {"source.csv": "abc123"}},
        scenarios,
        vectors,
        summary,
        targeted,
        margins,
        results,
        invariants,
    )

    assert "all_evidence_low" in report
    assert "LOW->MODERATE: 1" in report
    assert "Observed simplex band transitions" in report
    assert "82/88 completed; 2 fusion errors; 4 no-usable-evidence" in report
    assert "Two-modality masks had band stability" in report
    assert "Threshold flips peaked in" in report
