"""Canonical portable assessment envelopes and hashes."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from rural_stroke_assist.offline.contracts import AssessmentEnvelope


def _jsonable(value: Any) -> Any:
    if isinstance(value, AssessmentEnvelope):
        return value.model_dump(mode="json")
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    return value


def canonical_json(value: Any) -> str:
    return json.dumps(_jsonable(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def canonical_assessment_hash(envelope: AssessmentEnvelope) -> str:
    return hashlib.sha256(canonical_json(envelope).encode("utf-8")).hexdigest()

