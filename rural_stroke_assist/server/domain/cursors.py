"""Opaque stable cursor pagination helpers."""

from __future__ import annotations

import base64
import json


def encode_cursor(created_at: str, resource_id: str) -> str:
    payload = json.dumps([created_at, resource_id], separators=(",", ":")).encode("utf-8")
    return base64.urlsafe_b64encode(payload).decode("ascii").rstrip("=")


def decode_cursor(value: str) -> tuple[str, str]:
    try:
        padded = value + "=" * (-len(value) % 4)
        created_at, resource_id = json.loads(base64.urlsafe_b64decode(padded.encode("ascii")))
        if not isinstance(created_at, str) or not isinstance(resource_id, str):
            raise ValueError
        return created_at, resource_id
    except (ValueError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError("Invalid cursor.") from exc
