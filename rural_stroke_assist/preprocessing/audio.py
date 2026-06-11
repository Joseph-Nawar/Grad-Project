from pathlib import Path
from typing import Optional

import librosa
import numpy as np


AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}

TORGO_LABEL_MAPPING = {
    "F_Con": "control",
    "M_Con": "control",
    "F_Dys": "dysarthric",
    "M_Dys": "dysarthric",
}

def list_audio_files(directory: Path) -> list[Path]:
    """Return all audio files under a directory."""
    return [
        path
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS
    ]


def is_audio_readable(audio_path: Path) -> bool:
    """
    Check whether an audio file can be loaded and contains samples.

    A file can be technically readable but still empty, so we require
    a non-empty signal.
    """
    try:
        signal, _sample_rate = librosa.load(
            audio_path,
            sr=None,
            mono=True,
            duration=1.0,
        )

        return signal is not None and len(signal) > 0

    except Exception:
        return False


def get_audio_metadata(audio_path: Path) -> dict:
    """
    Extract basic audio metadata.

    Returns:
    - duration_seconds
    - sample_rate
    - number of samples
    - is_empty_audio
    """
    signal, sample_rate = librosa.load(audio_path, sr=None, mono=True)

    is_empty_audio = len(signal) == 0

    return {
        "duration_seconds": round(librosa.get_duration(y=signal, sr=sample_rate), 3)
        if not is_empty_audio
        else 0.0,
        "sample_rate": sample_rate,
        "num_samples": len(signal),
        "is_empty_audio": is_empty_audio,
    }


def infer_speech_label_from_path(audio_path: Path) -> str:
    """
    Infer TORGO speech label from the dataset's top-level folder.

    TORGO folder convention:
    - F_Con: female control speakers
    - M_Con: male control speakers
    - F_Dys: female dysarthric speakers
    - M_Dys: male dysarthric speakers

    This is intentionally schema-based rather than keyword-based to avoid
    fragile substring matching.
    """
    path_parts = audio_path.parts

    for folder_name, label in TORGO_LABEL_MAPPING.items():
        if folder_name in path_parts:
            return label

    return "unknown"

def infer_torgo_group_from_path(audio_path: Path) -> str:
    """
    Infer TORGO top-level group folder from path.

    Returns one of:
    - F_Con
    - M_Con
    - F_Dys
    - M_Dys
    - unknown
    """
    for folder_name in TORGO_LABEL_MAPPING:
        if folder_name in audio_path.parts:
            return folder_name

    return "unknown"


def infer_torgo_speaker_from_path(audio_path: Path) -> str:
    """
    Infer TORGO speaker ID from path.

    Expected examples:
    - FC01
    - MC01
    - F01
    - M01
    """
    group = infer_torgo_group_from_path(audio_path)

    if group == "unknown":
        return "unknown"

    parts = list(audio_path.parts)

    try:
        group_index = parts.index(group)
        return parts[group_index + 1]
    except (ValueError, IndexError):
        return "unknown"