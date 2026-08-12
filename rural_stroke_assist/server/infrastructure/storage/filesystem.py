"""Managed filesystem attachment storage for Stage 2."""

from __future__ import annotations

import hashlib
import os
import tempfile
from pathlib import Path
from typing import BinaryIO
from uuid import UUID, uuid4

from rural_stroke_assist.server.infrastructure.storage.protocol import StoredAttachment


class FilesystemAttachmentStore:
    ALLOWED = {
        "face": {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png"},
        "audio": {".wav": "audio/wav", ".mp3": "audio/mpeg", ".ogg": "audio/ogg"},
    }

    def __init__(self, root: str | Path, *, max_bytes: int = 25 * 1024 * 1024) -> None:
        self.root = Path(root).resolve()
        self.max_bytes = max_bytes
        self.root.mkdir(parents=True, exist_ok=True)

    def save(self, case_id: UUID | str, kind: str, stream: BinaryIO, *, filename: str | None, media_type: str, attachment_id: UUID | None = None) -> StoredAttachment:
        if kind not in self.ALLOWED:
            raise ValueError("Unsupported media kind.")
        extension = Path(filename or "").suffix.lower()
        if extension not in self.ALLOWED[kind] or self.ALLOWED[kind][extension] != media_type:
            raise ValueError("Unsupported media type or extension.")
        safe_case_id = str(case_id)
        case_dir = self.root / "cases" / safe_case_id
        case_dir.mkdir(parents=True, exist_ok=True)
        attachment_id = attachment_id or uuid4()
        storage_key = f"cases/{safe_case_id}/{attachment_id}{extension}"
        destination = self.root / storage_key
        if self.root not in destination.resolve().parents:
            raise ValueError("Unsafe attachment path.")
        digest = hashlib.sha256()
        size = 0
        temporary: str | None = None
        fd, temporary = tempfile.mkstemp(prefix=".upload-", dir=case_dir)
        try:
            with os.fdopen(fd, "wb") as output:
                while True:
                    chunk = stream.read(1024 * 1024)
                    if not chunk:
                        break
                    size += len(chunk)
                    if size > self.max_bytes:
                        raise ValueError("Attachment exceeds configured size limit.")
                    digest.update(chunk)
                    output.write(chunk)
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary, destination)
            temporary = None
        finally:
            if temporary and os.path.exists(temporary):
                os.unlink(temporary)
        if size == 0:
            destination.unlink(missing_ok=True)
            raise ValueError("Attachment is empty.")
        return StoredAttachment(attachment_id, case_id, kind, media_type, storage_key, size, digest.hexdigest())

    def resolve(self, attachment: StoredAttachment) -> Path:
        path = (self.root / attachment.storage_key).resolve()
        if self.root not in path.parents or not path.is_file():
            raise ValueError("Attachment is outside managed storage or missing.")
        return path

    def delete(self, attachment: StoredAttachment) -> None:
        self.resolve(attachment).unlink(missing_ok=True)
