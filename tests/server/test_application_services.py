from __future__ import annotations

import pytest

from rural_stroke_assist.server.application.idempotency import require_idempotency_key


def test_idempotency_key_is_required_and_bounded() -> None:
    assert require_idempotency_key("request-1") == "request-1"
    with pytest.raises(Exception):
        require_idempotency_key(None)
    with pytest.raises(Exception):
        require_idempotency_key("x" * 256)
