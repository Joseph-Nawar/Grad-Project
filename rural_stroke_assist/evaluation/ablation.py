"""Engineering-only fusion weight sensitivity scenarios."""
from __future__ import annotations
from typing import Any

from rural_stroke_assist.modules.fusion_module import FusionInput, fuse_multimodal_scores


WEIGHT_SCENARIOS = {"canonical": {"face": .35, "speech": .30, "acute_symptoms": .25, "metadata_context": .10}, "balanced_acute": {"face": .30, "speech": .30, "acute_symptoms": .30, "metadata_context": .10}, "symptom_emphasis": {"face": .30, "speech": .25, "acute_symptoms": .35, "metadata_context": .10}}


def run_weight_sensitivity() -> dict[str, Any]:
    cases = {"all_high": FusionInput(.8, .8, .8, .8), "face_missing": FusionInput(None, .8, .8, .8), "metadata_only": FusionInput(None, None, None, .8), "urgent": FusionInput(0.1, 0.1, 0.95, 0.1, True)}
    output = {}
    for name, values in cases.items():
        output[name] = {}
        for scenario, weights in WEIGHT_SCENARIOS.items():
            result = fuse_multimodal_scores(values, weights=weights, moderate_threshold=.4, high_threshold=.7)
            output[name][scenario] = {"evidence_score": result.fused_score, "band": result.risk_band, "normalized_weights": result.normalized_weights_used, "contributions": result.modality_contributions}
    return {"scope": "sensitivity analysis; no alternative is claimed more accurate", "scenarios": output}
