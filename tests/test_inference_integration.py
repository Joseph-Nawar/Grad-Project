from pathlib import Path

import pandas as pd
import pytest

from rural_stroke_assist.inference.face_adapter import FaceAdapter
from rural_stroke_assist.inference.metadata_adapter import MetadataAdapter, MetadataInput
from rural_stroke_assist.inference.registry import BaselineRegistry
from rural_stroke_assist.inference.speech_adapter import SpeechAdapter


ROOT = Path(__file__).resolve().parents[1]
REGISTRY = BaselineRegistry.from_file(ROOT / "config" / "baseline_registry.json")


@pytest.mark.integration
def test_real_face_artifact_inference_on_manifest_input() -> None:
    manifest = pd.read_csv(REGISTRY.manifest_path("face_split"))
    image_path = Path(manifest.loc[manifest["split"] == "test", "path"].iloc[0])

    result = FaceAdapter(registry=REGISTRY).infer(image_path)

    assert result.available is True
    assert result.score is not None
    assert 0.0 <= result.score <= 1.0
    assert result.provenance == REGISTRY.component("face").path


@pytest.mark.integration
def test_real_speech_artifact_inference_on_manifest_input() -> None:
    manifest = pd.read_csv(REGISTRY.manifest_path("speech_split"))
    audio_path = Path(manifest.loc[manifest["split"] == "test", "path"].iloc[0])

    result = SpeechAdapter(registry=REGISTRY).infer(audio_path)

    assert result.available is True
    assert result.score is not None
    assert 0.0 <= result.score <= 1.0
    assert result.provenance == REGISTRY.component("speech").path


@pytest.mark.integration
def test_real_metadata_artifact_inference_on_manifest_input() -> None:
    manifest = pd.read_csv(REGISTRY.manifest_path("metadata_split"))
    row = manifest.loc[manifest["split"] == "test"].iloc[0]
    metadata = MetadataInput(
        age=row["age"],
        hypertension=row["hypertension"],
        heart_disease=row["heart_disease"],
        avg_glucose_level=row["avg_glucose_level"],
        bmi=row["bmi"],
        gender=row["gender"],
        ever_married=row["ever_married"],
        work_type=row["work_type"],
        Residence_type=row["Residence_type"],
        smoking_status=row["smoking_status"],
    )

    result = MetadataAdapter(registry=REGISTRY).infer(metadata)

    assert result.available is True
    assert result.score is not None
    assert 0.0 <= result.score <= 1.0
    assert result.confidence is None
    assert result.provenance == REGISTRY.component("metadata_context").path
