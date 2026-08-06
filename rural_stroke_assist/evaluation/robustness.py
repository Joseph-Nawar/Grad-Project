"""Deterministic robustness checks that do not claim clinical performance."""
from __future__ import annotations
from itertools import combinations
from typing import Any


MODALITIES = ("face", "speech", "metadata_context", "acute_symptoms")


def modality_combinations() -> tuple[tuple[str, ...], ...]:
    return tuple(combo for size in range(1, len(MODALITIES) + 1) for combo in combinations(MODALITIES, size))


def scenario_coverage() -> dict[str, Any]:
    combos = modality_combinations()
    return {"non_empty_combinations": [list(combo) for combo in combos], "count": len(combos), "no_usable_modalities": "insufficient_evidence", "scope": "engineering robustness and deterministic scenario consistency"}
