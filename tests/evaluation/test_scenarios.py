from rural_stroke_assist.evaluation.ablation import run_weight_sensitivity
from rural_stroke_assist.evaluation.claim_registry import default_claims
from rural_stroke_assist.evaluation.robustness import modality_combinations
from rural_stroke_assist.evaluation.symptom_evaluator import evaluate_symptoms


def test_all_nonempty_modality_combinations_are_declared():
    assert len(modality_combinations()) == 15


def test_symptom_evaluation_is_rule_verification():
    result = evaluate_symptoms()
    assert result["rule_verification_only"] is True
    assert result["scenario_count"] >= 7


def test_ablations_do_not_claim_accuracy():
    result = run_weight_sensitivity()
    assert "no alternative is claimed more accurate" in result["scope"]


def test_claim_registry_marks_clinical_accuracy_unsupported():
    claims = default_claims()
    assert any(c.status.value == "unsupported" and "diagnostic accuracy" in c.claim for c in claims)
