from __future__ import annotations

from rural_stroke_assist.offline.clock import ExponentialBackoff
from rural_stroke_assist.offline.http import FailureAction, classify_http_failure


def test_exponential_backoff_is_bounded_and_injectably_jittered() -> None:
    policy = ExponentialBackoff(base_seconds=2, maximum_seconds=10, jitter_ratio=0.25)

    assert policy.delay_seconds(attempt=1, random_value=0.0) == 1.5
    assert policy.delay_seconds(attempt=2, random_value=1.0) == 5.0
    assert policy.delay_seconds(attempt=20, random_value=1.0) == 10.0


def test_connectivity_failures_are_retryable_without_attempt_limit() -> None:
    failure = classify_http_failure(None, error_code="timeout")

    assert failure.action is FailureAction.RETRY
    assert failure.indefinite is True


def test_http_error_policy_distinguishes_refresh_conflict_and_dead_letter() -> None:
    assert classify_http_failure(401).action is FailureAction.REFRESH_AUTH
    assert classify_http_failure(429).action is FailureAction.RETRY
    assert classify_http_failure(503).action is FailureAction.RETRY
    assert classify_http_failure(409, error_code="idempotency_conflict").action is FailureAction.DEAD_LETTER
    assert classify_http_failure(409, error_code="conflict").action is FailureAction.CONFLICT
    assert classify_http_failure(422, error_code="validation_error").action is FailureAction.DEAD_LETTER
