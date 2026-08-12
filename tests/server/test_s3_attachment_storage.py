from __future__ import annotations

import hashlib
import io
from pathlib import Path
from uuid import UUID

import pytest

from rural_stroke_assist.server.infrastructure.storage.protocol import StoredAttachment
from rural_stroke_assist.server.infrastructure.storage.s3 import S3AttachmentStore


class _Body(io.BytesIO):
    def close(self) -> None:
        super().close()


class FakeS3Client:
    def __init__(self) -> None:
        self.objects: dict[tuple[str, str], dict[str, object]] = {}

    def upload_fileobj(self, fileobj, bucket: str, key: str, ExtraArgs: dict[str, str]) -> None:
        self.objects[(bucket, key)] = {
            "body": fileobj.read(),
            "content_type": ExtraArgs["ContentType"],
        }

    def get_object(self, *, Bucket: str, Key: str) -> dict[str, object]:
        item = self.objects[(Bucket, Key)]
        body = item["body"]
        assert isinstance(body, bytes)
        return {"Body": _Body(body), "ContentLength": len(body)}

    def delete_object(self, *, Bucket: str, Key: str) -> None:
        del self.objects[(Bucket, Key)]


def test_s3_store_preserves_checksum_and_materializes_existing_path_contract(tmp_path: Path) -> None:
    client = FakeS3Client()
    store = S3AttachmentStore(bucket="demo", client=client, temp_root=tmp_path)
    data = b"fake image bytes"

    stored = store.save(UUID(int=1), "face", io.BytesIO(data), filename="capture.jpg", media_type="image/jpeg")

    assert stored.storage_key.startswith("cases/00000000-0000-0000-0000-000000000001/")
    assert stored.size_bytes == len(data)
    assert stored.checksum_sha256 == hashlib.sha256(data).hexdigest()
    resolved = store.resolve(stored)
    assert resolved.read_bytes() == data
    store.delete(stored)
    assert ("demo", stored.storage_key) not in client.objects


def test_s3_store_rejects_invalid_media_and_oversized_upload(tmp_path: Path) -> None:
    store = S3AttachmentStore(bucket="demo", client=FakeS3Client(), max_bytes=4, temp_root=tmp_path)

    with pytest.raises(ValueError, match="Unsupported media"):
        store.save("case", "face", io.BytesIO(b"x"), filename="capture.gif", media_type="image/gif")
    with pytest.raises(ValueError, match="exceeds"):
        store.save("case", "face", io.BytesIO(b"12345"), filename="capture.jpg", media_type="image/jpeg")


def test_s3_store_rejects_checksum_mismatch_on_materialization(tmp_path: Path) -> None:
    client = FakeS3Client()
    store = S3AttachmentStore(bucket="demo", client=client, temp_root=tmp_path)
    stored = store.save("case", "audio", io.BytesIO(b"audio"), filename="recording.wav", media_type="audio/wav")
    tampered = StoredAttachment(stored.id, stored.case_id, stored.kind, stored.media_type, stored.storage_key, stored.size_bytes, "0" * 64)

    with pytest.raises(ValueError, match="checksum"):
        store.resolve(tampered)


def test_s3_store_rejects_unsafe_object_keys(tmp_path: Path) -> None:
    store = S3AttachmentStore(bucket="demo", client=FakeS3Client(), temp_root=tmp_path)
    attachment = StoredAttachment(UUID(int=1), "case", "face", "image/jpeg", "../outside.jpg", 1, "0" * 64)

    with pytest.raises(ValueError, match="Unsafe"):
        store.resolve(attachment)
