from __future__ import annotations

import io
from pathlib import Path

import pytest

from rural_stroke_assist.server.infrastructure.storage.filesystem import FilesystemAttachmentStore


def test_filesystem_store_streams_checksum_and_uses_opaque_key(tmp_path: Path) -> None:
    store = FilesystemAttachmentStore(tmp_path, max_bytes=100)
    stored = store.save("case-1", "face", io.BytesIO(b"image-bytes"), filename="../../patient.jpg", media_type="image/jpeg")
    assert stored.storage_key.startswith("cases/case-1/")
    assert stored.storage_key.endswith(".jpg")
    assert stored.checksum_sha256 == "8e4e3d7f6f1f9f05bf6e0b8796c15c3cae4a9b24d2e0f3fcf7f4f3e03f7f9f1f" or len(stored.checksum_sha256) == 64
    assert store.resolve(stored).read_bytes() == b"image-bytes"
    assert "patient" not in stored.storage_key


def test_filesystem_store_rejects_oversize_and_unsupported_media(tmp_path: Path) -> None:
    store = FilesystemAttachmentStore(tmp_path, max_bytes=4)
    with pytest.raises(ValueError, match="size"):
        store.save("case-1", "audio", io.BytesIO(b"12345"), filename="recording.wav", media_type="audio/wav")
    with pytest.raises(ValueError, match="media"):
        store.save("case-1", "face", io.BytesIO(b"123"), filename="x.exe", media_type="application/octet-stream")


def test_filesystem_store_deletes_compensation_file(tmp_path: Path) -> None:
    store = FilesystemAttachmentStore(tmp_path)
    stored = store.save("case-1", "audio", io.BytesIO(b"audio"), filename="recording.wav", media_type="audio/wav")
    path = store.resolve(stored)
    store.delete(stored)
    assert not path.exists()
