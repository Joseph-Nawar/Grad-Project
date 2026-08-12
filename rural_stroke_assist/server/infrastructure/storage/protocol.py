"""Typed attachment storage protocol."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Protocol
from uuid import UUID


@dataclass(frozen=True)
class StoredAttachment:
    id: UUID
    case_id: UUID | str
    kind: str
    media_type: str
    storage_key: str
    size_bytes: int
    checksum_sha256: str


class AttachmentStore(Protocol):
    def save(self, case_id: UUID | str, kind: str, stream: BinaryIO, *, filename: str | None, media_type: str, attachment_id: UUID | None = None) -> StoredAttachment: ...
    def resolve(self, attachment: StoredAttachment) -> Path: ...
    def delete(self, attachment: StoredAttachment) -> None: ...
