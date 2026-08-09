"""Review application rules."""

from __future__ import annotations

from rural_stroke_assist.server.errors import ApiError


def validate_review_decision(agree: bool, alternative_disposition: str | None) -> None:
    if not agree and not alternative_disposition:
        raise ApiError("validation_error", "An override requires an alternative disposition.", status_code=422)
