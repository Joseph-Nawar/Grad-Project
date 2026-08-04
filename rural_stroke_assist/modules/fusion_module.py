from dataclasses import dataclass, field


DEFAULT_FUSION_WEIGHTS = {
    "face": 0.35,
    "speech": 0.30,
    "acute_symptoms": 0.25,
    "metadata_context": 0.10,
}

DEFAULT_MODERATE_THRESHOLD = 0.50
DEFAULT_HIGH_THRESHOLD = 0.75


@dataclass(frozen=True)
class FusionInput:
    """
    Normalized modality scores for fusion.

    All available scores should be in [0, 1].

    Missing modalities may be passed as None.
    """

    face_acute_score: float | None = None
    speech_acute_score: float | None = None
    acute_symptom_score: float | None = None
    metadata_contextual_risk_score: float | None = None

    acute_symptom_hard_escalation: bool = False


@dataclass(frozen=True)
class FusionResult:
    """
    Final multimodal fusion output.
    """

    fused_score: float
    risk_band: str
    modality_contributions: dict[str, float]
    normalized_weights_used: dict[str, float]
    evidence: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def _validate_score(name: str, value: float | None) -> None:
    """Validate that a score is either None or inside [0, 1]."""
    if value is None:
        return

    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be in [0, 1]. Got {value}.")


def _normalize_available_weights(
    available_scores: dict[str, float],
    weights: dict[str, float],
) -> dict[str, float]:
    """
    Renormalize weights over available modalities only.

    This allows fusion to still work when a modality is missing.
    """
    available_weight_sum = sum(weights[name] for name in available_scores)

    if available_weight_sum <= 0:
        raise ValueError("At least one modality score must be available.")

    return {
        name: weights[name] / available_weight_sum
        for name in available_scores
    }


def _validate_thresholds(
    moderate_threshold: float,
    high_threshold: float,
) -> None:
    """Validate ordinal band thresholds."""
    if not 0.0 <= moderate_threshold <= 1.0:
        raise ValueError(
            f"moderate_threshold must be in [0, 1]. Got {moderate_threshold}."
        )
    if not 0.0 <= high_threshold <= 1.0:
        raise ValueError(
            f"high_threshold must be in [0, 1]. Got {high_threshold}."
        )
    if moderate_threshold > high_threshold:
        raise ValueError(
            "moderate_threshold must be less than or equal to high_threshold."
        )


def _apply_acute_evidence_floor(
    fused_score: float,
    fusion_input: FusionInput,
    *,
    moderate_threshold: float,
    high_threshold: float,
) -> tuple[float, list[str]]:
    """
    Apply interpretable floors when multiple acute evidence sources corroborate.

    This keeps metadata as a weak background signal while ensuring that
    concordant acute evidence is not overly diluted by weighted averaging.
    """
    notes: list[str] = []

    acute_scores = {
        "face": fusion_input.face_acute_score,
        "speech": fusion_input.speech_acute_score,
        "acute_symptoms": fusion_input.acute_symptom_score,
    }
    present_acute_scores = {
        name: score for name, score in acute_scores.items() if score is not None
    }

    if len(present_acute_scores) < 2:
        return fused_score, notes

    face_score = fusion_input.face_acute_score
    speech_score = fusion_input.speech_acute_score
    symptom_score = fusion_input.acute_symptom_score

    high_corroboration = False
    if face_score is not None and speech_score is not None:
        high_corroboration = (
            face_score >= 0.75 and speech_score >= 0.75
        )

    if symptom_score is not None:
        high_corroboration = high_corroboration or (
            symptom_score >= 0.70
            and (
                (face_score is not None and face_score >= 0.75)
                or (speech_score is not None and speech_score >= 0.75)
            )
        )

    if sum(score >= 0.60 for score in present_acute_scores.values()) >= 3:
        high_corroboration = True

    if high_corroboration and fused_score < high_threshold:
        notes.append(
            "Acute corroboration floor applied: multiple acute evidence sources "
            "were strongly elevated, so the fusion score was raised to the HIGH threshold."
        )
        return high_threshold, notes

    moderate_corroboration = (
        sum(score >= 0.55 for score in present_acute_scores.values()) >= 2
    )
    if not moderate_corroboration:
        sorted_scores = sorted(present_acute_scores.values(), reverse=True)
        moderate_corroboration = (
            sorted_scores[0] >= 0.85 and sorted_scores[1] >= 0.35
        )

    if moderate_corroboration and fused_score < moderate_threshold:
        notes.append(
            "Acute corroboration floor applied: more than one acute evidence source "
            "was meaningfully elevated, so the fusion score was raised to the MODERATE threshold."
        )
        return moderate_threshold, notes

    return fused_score, notes


def assign_fusion_risk_band(
    fused_score: float,
    hard_escalation: bool = False,
    moderate_threshold: float = DEFAULT_MODERATE_THRESHOLD,
    high_threshold: float = DEFAULT_HIGH_THRESHOLD,
) -> str:
    """
    Convert fused score into an ordinal screening band.

    hard_escalation forces URGENT when the acute symptom module identifies
    multiple core FAST signs.
    """
    _validate_thresholds(
        moderate_threshold=moderate_threshold,
        high_threshold=high_threshold,
    )

    if hard_escalation:
        return "URGENT"

    if fused_score >= high_threshold:
        return "HIGH"

    if fused_score >= moderate_threshold:
        return "MODERATE"

    return "LOW"


def fuse_multimodal_scores(
    fusion_input: FusionInput,
    weights: dict[str, float] | None = None,
    moderate_threshold: float = DEFAULT_MODERATE_THRESHOLD,
    high_threshold: float = DEFAULT_HIGH_THRESHOLD,
) -> FusionResult:
    """
    Fuse available modality scores using interpretable weighted late fusion.

    This is an MVP fusion operator, not a learned fusion model.
    """
    weights = weights or DEFAULT_FUSION_WEIGHTS
    _validate_thresholds(
        moderate_threshold=moderate_threshold,
        high_threshold=high_threshold,
    )

    _validate_score("face_acute_score", fusion_input.face_acute_score)
    _validate_score("speech_acute_score", fusion_input.speech_acute_score)
    _validate_score("acute_symptom_score", fusion_input.acute_symptom_score)
    _validate_score(
        "metadata_contextual_risk_score",
        fusion_input.metadata_contextual_risk_score,
    )

    available_scores = {}

    if fusion_input.face_acute_score is not None:
        available_scores["face"] = fusion_input.face_acute_score

    if fusion_input.speech_acute_score is not None:
        available_scores["speech"] = fusion_input.speech_acute_score

    if fusion_input.acute_symptom_score is not None:
        available_scores["acute_symptoms"] = fusion_input.acute_symptom_score

    if fusion_input.metadata_contextual_risk_score is not None:
        available_scores["metadata_context"] = (
            fusion_input.metadata_contextual_risk_score
        )

    normalized_weights = _normalize_available_weights(
        available_scores=available_scores,
        weights=weights,
    )

    modality_contributions = {
        name: available_scores[name] * normalized_weights[name]
        for name in available_scores
    }

    fused_score = sum(modality_contributions.values())

    if fusion_input.acute_symptom_hard_escalation:
        fused_score = max(fused_score, 0.85)
        acute_floor_notes: list[str] = []
    else:
        fused_score, acute_floor_notes = _apply_acute_evidence_floor(
            fused_score=fused_score,
            fusion_input=fusion_input,
            moderate_threshold=moderate_threshold,
            high_threshold=high_threshold,
        )

    risk_band = assign_fusion_risk_band(
        fused_score=fused_score,
        hard_escalation=fusion_input.acute_symptom_hard_escalation,
        moderate_threshold=moderate_threshold,
        high_threshold=high_threshold,
    )

    evidence = [
        f"{name}: score={available_scores[name]:.3f}, "
        f"weight={normalized_weights[name]:.3f}, "
        f"contribution={modality_contributions[name]:.3f}"
        for name in available_scores
    ]
    evidence.extend(acute_floor_notes)

    warnings = [
        "Fusion output is a research screening estimate, not a clinical diagnosis.",
        "Metadata context is treated as background risk, not acute stroke evidence.",
    ]

    missing_modalities = set(weights) - set(available_scores)

    if missing_modalities:
        warnings.append(
            "Missing modalities were excluded and remaining weights were renormalized: "
            + ", ".join(sorted(missing_modalities))
        )

    if fusion_input.acute_symptom_hard_escalation:
        warnings.append(
            "Acute symptom module triggered hard escalation due to multiple FAST signs."
        )

    return FusionResult(
        fused_score=fused_score,
        risk_band=risk_band,
        modality_contributions=modality_contributions,
        normalized_weights_used=normalized_weights,
        evidence=evidence,
        warnings=warnings,
    )
