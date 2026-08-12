"""S3-compatible attachment storage for MinIO and AWS S3."""

from __future__ import annotations

import hashlib
import os
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO
from uuid import UUID, uuid4

from rural_stroke_assist.server.infrastructure.storage.protocol import StoredAttachment


class S3AttachmentStore:
    """Store immutable attachments in an S3-compatible object bucket.

    The existing Stage 2 callers require a local ``Path`` for inference and
    ``FileResponse``. ``resolve`` therefore downloads a checksum-verified
    private temporary copy while preserving the existing store protocol.
    """

    ALLOWED = {
        "face": {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png"},
        "audio": {".wav": "audio/wav", ".mp3": "audio/mpeg", ".ogg": "audio/ogg"},
    }

    def __init__(
        self,
        *,
        bucket: str,
        region: str | None = None,
        endpoint_url: str | None = None,
        access_key_id: str | None = None,
        secret_access_key: str | None = None,
        client: Any | None = None,
        temp_root: str | Path | None = None,
        max_bytes: int = 25 * 1024 * 1024,
    ) -> None:
        if not bucket.strip():
            raise ValueError("S3 bucket is required.")
        self.bucket = bucket
        self.max_bytes = max_bytes
        self._client = client or self._create_client(
            region=region,
            endpoint_url=endpoint_url,
            access_key_id=access_key_id,
            secret_access_key=secret_access_key,
        )
        self._temp_root = Path(temp_root or tempfile.mkdtemp(prefix="ruralstroke-s3-"))
        self._temp_root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _create_client(
        *,
        region: str | None,
        endpoint_url: str | None,
        access_key_id: str | None,
        secret_access_key: str | None,
    ) -> Any:
        try:
            import boto3
        except ImportError as exc:  # pragma: no cover - exercised by image build/import checks
            raise RuntimeError("boto3 is required for S3 attachment storage.") from exc
        kwargs: dict[str, Any] = {"region_name": region, "endpoint_url": endpoint_url}
        if access_key_id:
            kwargs["aws_access_key_id"] = access_key_id
        if secret_access_key:
            kwargs["aws_secret_access_key"] = secret_access_key
        return boto3.client("s3", **kwargs)

    @staticmethod
    def _validate_key(key: str) -> None:
        path = PurePosixPath(key)
        if path.is_absolute() or ".." in path.parts or "\\" in key or not key.startswith("cases/"):
            raise ValueError("Unsafe attachment storage key.")

    @classmethod
    def _validate_upload(cls, kind: str, filename: str | None, media_type: str) -> str:
        if kind not in cls.ALLOWED:
            raise ValueError("Unsupported media kind.")
        extension = Path(filename or "").suffix.lower()
        if extension not in cls.ALLOWED[kind] or cls.ALLOWED[kind][extension] != media_type:
            raise ValueError("Unsupported media type or extension.")
        return extension

    @staticmethod
    def _case_segment(case_id: UUID | str) -> str:
        segment = str(case_id)
        if not segment or "/" in segment or "\\" in segment or segment in {".", ".."}:
            raise ValueError("Unsafe case identifier.")
        return segment

    def save(
        self,
        case_id: UUID | str,
        kind: str,
        stream: BinaryIO,
        *,
        filename: str | None,
        media_type: str,
        attachment_id: UUID | None = None,
    ) -> StoredAttachment:
        extension = self._validate_upload(kind, filename, media_type)
        safe_case_id = self._case_segment(case_id)
        attachment_id = attachment_id or uuid4()
        storage_key = f"cases/{safe_case_id}/{attachment_id}{extension}"
        self._validate_key(storage_key)
        digest = hashlib.sha256()
        size = 0
        with tempfile.SpooledTemporaryFile(max_size=8 * 1024 * 1024, mode="w+b") as temporary:
            while True:
                chunk = stream.read(1024 * 1024)
                if not chunk:
                    break
                size += len(chunk)
                if size > self.max_bytes:
                    raise ValueError("Attachment exceeds configured size limit.")
                digest.update(chunk)
                temporary.write(chunk)
            if size == 0:
                raise ValueError("Attachment is empty.")
            temporary.seek(0)
            self._client.upload_fileobj(
                temporary,
                self.bucket,
                storage_key,
                ExtraArgs={"ContentType": media_type, "Metadata": {"sha256": digest.hexdigest()}},
            )
        return StoredAttachment(attachment_id, case_id, kind, media_type, storage_key, size, digest.hexdigest())

    def _temporary_path(self, storage_key: str) -> Path:
        return self._temp_root / f"{hashlib.sha256(storage_key.encode('utf-8')).hexdigest()}.bin"

    def resolve(self, attachment: StoredAttachment) -> Path:
        self._validate_key(attachment.storage_key)
        destination = self._temporary_path(attachment.storage_key)
        if destination.is_file():
            data = destination.read_bytes()
            if len(data) == attachment.size_bytes and hashlib.sha256(data).hexdigest() == attachment.checksum_sha256:
                return destination
            destination.unlink(missing_ok=True)

        response = self._client.get_object(Bucket=self.bucket, Key=attachment.storage_key)
        body = response["Body"]
        digest = hashlib.sha256()
        size = 0
        try:
            with destination.open("wb") as output:
                while True:
                    chunk = body.read(1024 * 1024)
                    if not chunk:
                        break
                    size += len(chunk)
                    if size > self.max_bytes:
                        raise ValueError("Attachment exceeds configured size limit.")
                    digest.update(chunk)
                    output.write(chunk)
        finally:
            close = getattr(body, "close", None)
            if close:
                close()
        if size != attachment.size_bytes or digest.hexdigest() != attachment.checksum_sha256:
            destination.unlink(missing_ok=True)
            raise ValueError("Attachment checksum verification failed.")
        return destination

    def delete(self, attachment: StoredAttachment) -> None:
        self._validate_key(attachment.storage_key)
        self._client.delete_object(Bucket=self.bucket, Key=attachment.storage_key)
        self._temporary_path(attachment.storage_key).unlink(missing_ok=True)
