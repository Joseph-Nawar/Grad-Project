from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from typing import Any

import jwt
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa

from rural_stroke_assist.server.infrastructure.auth.cognito import CognitoTokenVerifier


ISSUER = "https://cognito-idp.eu-central-1.amazonaws.com/eu-central-1_example"


def _key(kid: str) -> tuple[bytes, dict[str, Any]]:
    private = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    private_pem = private.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption())
    jwk = json.loads(jwt.algorithms.RSAAlgorithm.to_jwk(private.public_key()))
    jwk["kid"] = kid
    return private_pem, jwk


def _token(private_pem: bytes, kid: str, **overrides: Any) -> str:
    now = datetime.now(timezone.utc)
    payload: dict[str, Any] = {
        "sub": "user-1",
        "iss": ISSUER,
        "client_id": "client-1",
        "token_use": "access",
        "iat": now,
        "exp": now + timedelta(minutes=5),
        "cognito:groups": ["collector"],
        "facilities": ["facility-a"],
    }
    payload.update(overrides)
    return jwt.encode(payload, private_pem, algorithm="RS256", headers={"kid": kid})


def test_cognito_verifier_validates_signature_claims_groups_and_facilities() -> None:
    private_pem, jwk = _key("key-1")
    verifier = CognitoTokenVerifier(issuer=ISSUER, client_id="client-1", fetch_json=lambda _url: {"keys": [jwk]})

    principal = verifier.verify(_token(private_pem, "key-1"))

    assert principal.subject == "user-1"
    assert principal.roles == frozenset({"collector"})
    assert principal.facilities == frozenset({"facility-a"})


@pytest.mark.parametrize(
    "claims",
    [
        {"token_use": "id"},
        {"client_id": "wrong-client"},
        {"iss": "https://wrong.example"},
        {"exp": datetime.now(timezone.utc) - timedelta(minutes=1)},
    ],
)
def test_cognito_verifier_rejects_invalid_access_token_claims(claims: dict[str, Any]) -> None:
    private_pem, jwk = _key("key-1")
    verifier = CognitoTokenVerifier(issuer=ISSUER, client_id="client-1", fetch_json=lambda _url: {"keys": [jwk]})

    with pytest.raises(ValueError):
        verifier.verify(_token(private_pem, "key-1", **claims))


def test_cognito_verifier_refreshes_jwks_when_key_rotates() -> None:
    first_private, first_jwk = _key("key-1")
    second_private, second_jwk = _key("key-2")
    responses = iter(({"keys": [first_jwk]}, {"keys": [second_jwk]}))
    verifier = CognitoTokenVerifier(issuer=ISSUER, client_id="client-1", fetch_json=lambda _url: next(responses), cache_ttl_seconds=3600)

    verifier.verify(_token(first_private, "key-1"))
    rotated = verifier.verify(_token(second_private, "key-2"))

    assert rotated.subject == "user-1"
