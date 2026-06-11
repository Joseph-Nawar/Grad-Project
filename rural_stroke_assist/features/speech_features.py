from pathlib import Path

import librosa
import numpy as np
import pandas as pd


class EmptyAudioError(ValueError):
    """Raised when an audio file contains no usable samples."""

SPEECH_LABEL_MAPPING = {
    "control": 0,
    "dysarthric": 1,
}

SPEECH_INVERSE_LABEL_MAPPING = {
    0: "control",
    1: "dysarthric",
}


def load_audio(
    audio_path: str | Path,
    target_sample_rate: int = 16000,
    max_duration_seconds: float = 5.0,
) -> tuple[np.ndarray, int]:
    """
    Load audio as mono, resample, and optionally trim to fixed max duration.

    Raises:
        EmptyAudioError:
            If the loaded signal contains zero samples.
    """
    signal, sample_rate = librosa.load(
        audio_path,
        sr=target_sample_rate,
        mono=True,
        duration=max_duration_seconds,
    )

    if signal is None or len(signal) == 0:
        raise EmptyAudioError(f"Empty audio file: {audio_path}")

    return signal, sample_rate


def extract_mfcc_summary_features(
    signal: np.ndarray,
    sample_rate: int,
    n_mfcc: int = 13,
) -> dict:
    """
    Extract MFCC summary statistics.

    For each MFCC coefficient, compute mean and standard deviation.
    """
    mfcc = librosa.feature.mfcc(
        y=signal,
        sr=sample_rate,
        n_mfcc=n_mfcc,
    )

    features = {}

    mfcc_means = mfcc.mean(axis=1)
    mfcc_stds = mfcc.std(axis=1)

    for index, value in enumerate(mfcc_means, start=1):
        features[f"mfcc_{index}_mean"] = float(value)

    for index, value in enumerate(mfcc_stds, start=1):
        features[f"mfcc_{index}_std"] = float(value)

    return features


def extract_basic_audio_features(
    signal: np.ndarray,
    sample_rate: int,
) -> dict:
    """
    Extract simple acoustic features.
    """
    duration = librosa.get_duration(y=signal, sr=sample_rate)

    rms = librosa.feature.rms(y=signal)
    zero_crossing_rate = librosa.feature.zero_crossing_rate(signal)
    spectral_centroid = librosa.feature.spectral_centroid(y=signal, sr=sample_rate)
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=signal, sr=sample_rate)
    spectral_rolloff = librosa.feature.spectral_rolloff(y=signal, sr=sample_rate)

    return {
        "duration_seconds": float(duration),
        "rms_mean": float(rms.mean()),
        "rms_std": float(rms.std()),
        "zcr_mean": float(zero_crossing_rate.mean()),
        "zcr_std": float(zero_crossing_rate.std()),
        "spectral_centroid_mean": float(spectral_centroid.mean()),
        "spectral_centroid_std": float(spectral_centroid.std()),
        "spectral_bandwidth_mean": float(spectral_bandwidth.mean()),
        "spectral_bandwidth_std": float(spectral_bandwidth.std()),
        "spectral_rolloff_mean": float(spectral_rolloff.mean()),
        "spectral_rolloff_std": float(spectral_rolloff.std()),
    }


def extract_features_from_audio_file(
    audio_path: str | Path,
    target_sample_rate: int = 16000,
    max_duration_seconds: float = 5.0,
) -> dict:
    """
    Extract all MVP speech features from one audio file.
    """
    signal, sample_rate = load_audio(
        audio_path=audio_path,
        target_sample_rate=target_sample_rate,
        max_duration_seconds=max_duration_seconds,
    )

    features = {}
    features.update(
        extract_mfcc_summary_features(
            signal=signal,
            sample_rate=sample_rate,
            n_mfcc=13,
        )
    )
    features.update(
        extract_basic_audio_features(
            signal=signal,
            sample_rate=sample_rate,
        )
    )

    return features


def build_speech_features_table(
    split_manifest: pd.DataFrame,
    target_sample_rate: int = 16000,
    max_duration_seconds: float = 5.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build a tabular feature dataset from a speech split manifest.

    Returns:
    - features_df: successfully extracted audio features
    - failed_df: files that failed feature extraction
    """
    records = []
    failed_records = []

    for _, row in split_manifest.iterrows():
        try:
            features = extract_features_from_audio_file(
                audio_path=row["path"],
                target_sample_rate=target_sample_rate,
                max_duration_seconds=max_duration_seconds,
            )

            records.append(
                {
                    "path": row["path"],
                    "file_name": row["file_name"],
                    "label": row["label"],
                    "label_encoded": SPEECH_LABEL_MAPPING[row["label"]],
                    "speaker_id": row["speaker_id"],
                    "torgo_group": row["torgo_group"],
                    "split": row["split"],
                    **features,
                }
            )

        except Exception as exc:
            failed_records.append(
                {
                    "path": row.get("path"),
                    "file_name": row.get("file_name"),
                    "label": row.get("label"),
                    "speaker_id": row.get("speaker_id"),
                    "split": row.get("split"),
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                }
            )

    features_df = pd.DataFrame(records)
    failed_df = pd.DataFrame(failed_records)

    return features_df, failed_df