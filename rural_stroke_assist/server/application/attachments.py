"""Logical attachment identity and idempotency fingerprints."""

from __future__ import annotations

from rural_stroke_assist.server.domain.hashing import canonical_sha256


def attachment_request_fingerprint(*, case_id: str, attachment_id: str | None, kind: str, media_type: str, size_bytes: int, checksum_sha256: str) -> str:
    return canonical_sha256({"case_id": case_id, "attachment_id": attachment_id, "kind": kind, "media_type": media_type, "size_bytes": size_bytes, "checksum_sha256": checksum_sha256})

