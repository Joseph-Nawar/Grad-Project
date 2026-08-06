"""Managed, relative-path attachment storage."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from uuid import uuid4

from rural_stroke_assist.cases.contracts import AttachmentKind, AttachmentReference
from rural_stroke_assist.cases.exceptions import AttachmentValidationError


class AttachmentStore:
    ALLOWED = {
        AttachmentKind.FACE: {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png"},
        AttachmentKind.AUDIO: {".wav": "audio/wav", ".mp3": "audio/mpeg", ".ogg": "audio/ogg"},
    }

    def __init__(self, root: str | Path, *, max_bytes: int = 25 * 1024 * 1024) -> None:
        self.root = Path(root).resolve()
        self.max_bytes = max_bytes

    def save(self, case_id: str, kind: AttachmentKind, data: bytes, *, filename: str | None = None, media_type: str | None = None) -> AttachmentReference:
        if not data or len(data) > self.max_bytes:
            raise AttachmentValidationError("Attachment is empty or exceeds the configured size limit.")
        extension = Path(filename).suffix.lower() if filename else ""
        if not extension and media_type:
            extension = next((suffix for suffix, value in self.ALLOWED[kind].items() if value == media_type), "")
        extension = extension or {AttachmentKind.FACE: ".jpg", AttachmentKind.AUDIO: ".wav"}[kind]
        if extension not in self.ALLOWED[kind]:
            raise AttachmentValidationError("Attachment extension is not allowed.")
        expected_type = self.ALLOWED[kind][extension]
        if media_type and media_type not in set(self.ALLOWED[kind].values()):
            raise AttachmentValidationError("Attachment MIME type is not allowed.")
        if media_type and media_type != expected_type:
            raise AttachmentValidationError("Attachment extension and MIME type do not match.")
        case_dir = self.root / "cases" / case_id
        case_dir.mkdir(parents=True, exist_ok=True)
        safe_filename = f"{kind.value}-{uuid4().hex}{extension}"
        destination = case_dir / safe_filename
        if destination.parent != case_dir:
            raise AttachmentValidationError("Unsafe attachment path.")
        fd, temporary = tempfile.mkstemp(prefix=".upload-", dir=case_dir)
        try:
            with os.fdopen(fd, "wb") as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, destination)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        return AttachmentReference(kind, str(destination.relative_to(self.root)).replace("\\", "/"), media_type or expected_type, len(data))

    def resolve(self, reference: AttachmentReference) -> Path:
        path = (self.root / reference.relative_path).resolve()
        if self.root not in path.parents or not path.is_file():
            raise AttachmentValidationError("Attachment reference is outside managed storage or missing.")
        return path

    def delete_case(self, case_id: str) -> None:
        case_dir = (self.root / "cases" / case_id).resolve()
        if self.root not in case_dir.parents:
            raise AttachmentValidationError("Unsafe attachment cleanup path.")
        if case_dir.is_dir():
            for child in case_dir.iterdir():
                if child.is_file():
                    child.unlink()
            case_dir.rmdir()
