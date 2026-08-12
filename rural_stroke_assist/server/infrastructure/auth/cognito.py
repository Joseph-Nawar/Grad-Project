"""Cognito access-token verification with cached JWKS rotation support."""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Callable, Iterable, Mapping
from typing import Any
from urllib.request import urlopen

import jwt

from rural_stroke_assist.server.principal import Principal


JsonFetcher = Callable[[str], Mapping[str, Any]]


def _fetch_json(url: str) -> Mapping[str, Any]:
    with urlopen(url, timeout=5) as response:
        payload = json.loads(response.read().decode("utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("JWKS response must be a JSON object.")
    return payload


class CognitoTokenVerifier:
    """Verify RS256 Cognito access tokens and map claims to ``Principal``."""

    def __init__(
        self,
        *,
        issuer: str,
        client_id: str | None = None,
        client_ids: Iterable[str] | None = None,
        jwks_uri: str | None = None,
        fetch_json: JsonFetcher | None = None,
        cache_ttl_seconds: int = 900,
    ) -> None:
        accepted_client_ids = frozenset(client_ids or (() if client_id is None else (client_id,)))
        if not issuer or not accepted_client_ids:
            raise ValueError("Cognito issuer and client ID are required.")
        self.issuer = issuer.rstrip("/")
        self.client_ids = accepted_client_ids
        self.jwks_uri = jwks_uri or f"{self.issuer}/.well-known/jwks.json"
        self._fetch_json = fetch_json or _fetch_json
        self._cache_ttl_seconds = cache_ttl_seconds
        self._keys: dict[str, Any] = {}
        self._loaded_at = 0.0
        self._lock = threading.RLock()

    def _refresh(self) -> dict[str, Any]:
        document = self._fetch_json(self.jwks_uri)
        raw_keys = document.get("keys")
        if not isinstance(raw_keys, list):
            raise ValueError("Cognito JWKS response does not contain keys.")
        keys: dict[str, Any] = {}
        for jwk in raw_keys:
            if not isinstance(jwk, Mapping) or jwk.get("kid") is None:
                continue
            if jwk.get("kty") != "RSA" or jwk.get("alg", "RS256") != "RS256":
                continue
            try:
                keys[str(jwk["kid"])] = jwt.algorithms.RSAAlgorithm.from_jwk(json.dumps(dict(jwk)))
            except (TypeError, ValueError, KeyError):
                continue
        if not keys:
            raise ValueError("Cognito JWKS response has no usable RS256 keys.")
        with self._lock:
            self._keys = keys
            self._loaded_at = time.monotonic()
        return keys

    def _cached_keys(self) -> dict[str, Any]:
        with self._lock:
            fresh = self._keys and time.monotonic() - self._loaded_at < self._cache_ttl_seconds
            if fresh:
                return dict(self._keys)
        return self._refresh()

    @staticmethod
    def _claim_list(payload: Mapping[str, Any], *names: str) -> list[str]:
        for name in names:
            value = payload.get(name)
            if value is None:
                continue
            if isinstance(value, str):
                try:
                    decoded = json.loads(value)
                except json.JSONDecodeError:
                    decoded = [item.strip() for item in value.split(",") if item.strip()]
                value = decoded
            if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
                raise ValueError(f"Cognito claim {name} must be a list of strings.")
            return value
        return []

    def verify(self, token: str) -> Principal:
        try:
            header = jwt.get_unverified_header(token)
            if header.get("alg") != "RS256" or not header.get("kid"):
                raise ValueError("Unsupported Cognito token algorithm or key ID.")
            kid = str(header["kid"])
            keys = self._cached_keys()
            key = keys.get(kid)
            if key is None:
                keys = self._refresh()
                key = keys.get(kid)
            if key is None:
                raise ValueError("Cognito signing key is unknown.")
            payload: dict[str, Any] = jwt.decode(
                token,
                key,
                algorithms=["RS256"],
                issuer=self.issuer,
                options={"verify_aud": False},
            )
            if payload.get("token_use") != "access":
                raise ValueError("Cognito access token is required.")
            if payload.get("client_id") not in self.client_ids:
                raise ValueError("Cognito token client ID is invalid.")
            if payload.get("aud") is not None and payload.get("aud") not in self.client_ids:
                raise ValueError("Cognito token audience is invalid.")
            subject = payload.get("sub")
            if not isinstance(subject, str) or not subject.strip():
                raise ValueError("Cognito token subject is required.")
            groups = self._claim_list(payload, "cognito:groups")
            facilities = self._claim_list(payload, "facilities", "custom:facilities", "custom:facility_scope")
            return Principal(subject, groups, facilities)
        except ValueError:
            raise
        except Exception as exc:
            raise ValueError("Invalid Cognito access token.") from exc
