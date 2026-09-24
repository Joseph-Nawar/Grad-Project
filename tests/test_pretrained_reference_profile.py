from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from rural_stroke_assist.assessment.factory import create_default_assessment_service
from rural_stroke_assist.inference.exceptions import ArtifactConfigurationError
from rural_stroke_assist.inference.reference_adapters import (
    ReferenceMetadataAdapter,
    ReferenceSpeechAdapter,
)
from rural_stroke_assist.inference.reference_registry import (
    ReferenceProfileRegistry,
    sha256_file,
    training_context_fingerprint,
)
from rural_stroke_assist.inference.reference_runners import (
    DistilHuBERTRunner,
    TabPFNRunner,
)
from rural_stroke_assist.inference.contracts import QualityStatus
from rural_stroke_assist.inference.metadata_adapter import MetadataInput
from rural_stroke_assist.quality.audio_quality import AudioQualityAssessment


ROOT = Path(__file__).resolve().parents[1]
REFERENCE_CONFIG = ROOT / "config" / "pretrained_reference_registry.json"


def test_reference_registry_requires_project_owned_artifacts(tmp_path: Path) -> None:
    config = tmp_path / "registry.json"
    config.write_text(
        """{
          "schema_version": 1,
          "runtime_profile": "pretrained_reference",
          "root": ".",
          "modalities": {
            "speech": {"artifact": "missing.joblib", "artifact_sha256": "abc"}
          }
        }
        """,
        encoding="utf-8",
    )

    registry = ReferenceProfileRegistry.from_file(config)

    with pytest.raises(ArtifactConfigurationError, match="speech.*missing"):
        registry.validate_project_artifacts()


def test_training_context_fingerprint_is_stable_for_frozen_rows() -> None:
    frame = pd.DataFrame(
        {
            "age": pd.Series([40.0], dtype="float64"),
            "hypertension": pd.Series([0.0], dtype="float64"),
            "heart_disease": pd.Series([0.0], dtype="float64"),
            "avg_glucose_level": pd.Series([90.0], dtype="float64"),
            "bmi": pd.Series([25.0], dtype="float64"),
            "gender": pd.Series(["Female"], dtype="object"),
            "ever_married": pd.Series(["No"], dtype="object"),
            "work_type": pd.Series(["Private"], dtype="object"),
            "Residence_type": pd.Series(["Rural"], dtype="object"),
            "smoking_status": pd.Series(["never smoked"], dtype="object"),
            "stroke": pd.Series([0], dtype="int64"),
            "split": pd.Series(["train"], dtype="object"),
        }
    )

    first = training_context_fingerprint(frame)
    second = training_context_fingerprint(frame.copy())

    assert first == second
    assert first["sha256"]
    assert first["row_count"] == 1


def test_reference_registry_verifies_classifier_hash(tmp_path: Path) -> None:
    artifact = tmp_path / "classifier.joblib"
    artifact.write_bytes(b"selected-classifier")
    expected = sha256_file(artifact)
    config = tmp_path / "registry.json"
    config.write_text(
        f"""{{
          "schema_version": 1,
          "runtime_profile": "pretrained_reference",
          "root": ".",
          "modalities": {{
            "speech": {{"artifact": "classifier.joblib", "artifact_sha256": "{expected}"}}
          }}
        }}
        """,
        encoding="utf-8",
    )
    registry = ReferenceProfileRegistry.from_file(config)
    registry.validate_project_artifacts()

    artifact.write_bytes(b"tampered")
    with pytest.raises(ArtifactConfigurationError, match="hash mismatch"):
        registry.validate_project_artifacts()


def test_reference_profile_is_explicit_and_selects_reference_runners(monkeypatch) -> None:
    monkeypatch.delenv("RURALSTROKE_EDGE_RUNTIME_PROFILE", raising=False)
    import rural_stroke_assist.inference.reference_runtime as reference_runtime

    registry = object()
    runners = {"face": object(), "speech": object(), "metadata_context": object()}
    adapters = {
        "face": object(),
        "speech": object(),
        "metadata_context": object(),
        "acute_symptoms": object(),
    }
    monkeypatch.setattr(
        reference_runtime,
        "build_pretrained_reference_runtime",
        lambda: (registry, runners),
    )
    monkeypatch.setattr(
        reference_runtime,
        "build_pretrained_reference_adapters",
        lambda selected_registry, selected_runners: (
            adapters
            if selected_registry is registry and selected_runners is runners
            else None
        ),
    )

    service = create_default_assessment_service(profile="pretrained_reference")

    assert service.runtime_profile == "pretrained_reference"
    assert service.adapters == adapters


def test_reference_speech_adapter_preserves_dysarthria_semantics(tmp_path: Path) -> None:
    audio = tmp_path / "voice.wav"
    audio.write_bytes(b"fixture")
    signal = np.ones(16_000, dtype=np.float32) * 0.05

    class Runner:
        def run(self, value: np.ndarray):
            assert value.shape == (16_000,)
            return np.asarray([[0.2, 0.8]]), ("control", "dysarthric")

    class Registry:
        def component(self, _name):
            return type(
                "Component",
                (),
                {
                    "data": {
                        "class_mapping": {"0": "control", "1": "dysarthric"},
                        "thresholds": {"classification": 0.5},
                    },
                    "path": "models/reference/speech_distilhubert_logistic_regression.joblib",
                    "provenance": "profile=pretrained_reference; model=distilhubert",
                },
            )()

    adapter = ReferenceSpeechAdapter(
        registry=Registry(),
        runner=Runner(),
        audio_loader=lambda _path: (signal, 16_000),
        quality_assessor=lambda _signal, _rate: AudioQualityAssessment(QualityStatus.PASS, ()),
    )
    result = adapter.infer(audio)

    assert result.score == pytest.approx(0.8)
    assert result.score_semantics == "dysarthria_proxy_evidence"
    assert "dysarthria proxy" in " ".join(result.warnings).lower()
    assert "pretrained_reference" in result.provenance


def test_reference_metadata_adapter_preserves_contextual_semantics() -> None:
    class Runner:
        def run(self, frame: pd.DataFrame):
            assert list(frame.columns) == [
                "age", "hypertension", "heart_disease", "avg_glucose_level", "bmi",
                "gender", "ever_married", "work_type", "Residence_type", "smoking_status",
            ]
            return np.asarray([[0.9, 0.1]]), (0, 1)

    class Registry:
        def component(self, _name):
            return type(
                "Component",
                (),
                {
                    "data": {
                        "class_mapping": {"0": "no_stroke", "1": "stroke"},
                        "feature_columns": [
                            "age", "hypertension", "heart_disease", "avg_glucose_level", "bmi",
                            "gender", "ever_married", "work_type", "Residence_type", "smoking_status",
                        ],
                        "thresholds": {"classification": 0.5},
                    },
                    "path": "models/reference/tabpfn_v2_training_context.csv",
                    "provenance": "profile=pretrained_reference; model=tabpfn_v2",
                },
            )()

    adapter = ReferenceMetadataAdapter(registry=Registry(), runner=Runner())
    result = adapter.infer(
        MetadataInput(
            age=68, hypertension=1, heart_disease=0, avg_glucose_level=130.0, bmi=25.0,
            gender="Male", ever_married="Yes", work_type="Private", residence_type="Urban",
            smoking_status="never smoked",
        )
    )

    assert result.score == pytest.approx(0.1)
    assert result.score_semantics == "contextual_risk_evidence"
    assert "contextual" in " ".join(result.warnings).lower()
    assert "pretrained_reference" in result.provenance


def test_reference_runner_constructors_are_real_components() -> None:
    assert DistilHuBERTRunner.__name__ == "DistilHuBERTRunner"
    assert TabPFNRunner.__name__ == "TabPFNRunner"


def test_distilhubert_runner_reproduces_mean_pooling_and_classifier_identity(
    tmp_path: Path,
) -> None:
    classifier_path = tmp_path / "classifier.joblib"
    classifier_path.write_bytes(b"classifier")
    captured: dict[str, object] = {}

    class Tensor:
        def __init__(self, value):
            self.value = value

        def detach(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return self.value

        def __getitem__(self, index):
            return Tensor(self.value[index])

    class Torch:
        class _Context:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

        def inference_mode(self):
            return self._Context()

    class Extractor:
        def __call__(self, signal, **kwargs):
            captured["signal"] = signal
            captured["kwargs"] = kwargs
            return {"input_values": object()}

    class Encoder:
        def eval(self):
            return self

        def __call__(self, **_kwargs):
            return SimpleNamespace(
                last_hidden_state=Tensor(np.asarray([[[1.0, 2.0], [3.0, 4.0]]]))
            )

    class Classifier:
        classes_ = np.asarray([0, 1])

        def predict_proba(self, value):
            captured["embedding"] = value
            return np.asarray([[0.25, 0.75]])

    class Registry:
        def validate_project_artifacts(self):
            return None

        def resolve_speech_snapshot(self, *, local_only):
            assert local_only is True
            return tmp_path

        def path_for(self, name):
            assert name == "speech"
            return classifier_path

    runner = DistilHuBERTRunner(
        Registry(),
        snapshot_path=tmp_path,
        encoder_loader=lambda _path: (Extractor(), Encoder(), Torch()),
        classifier_loader=lambda _path: Classifier(),
    )
    probabilities, classes = runner.run(np.ones(16_000, dtype=np.float32))

    assert np.array_equal(probabilities, np.asarray([[0.25, 0.75]]))
    assert classes == (0, 1)
    assert np.array_equal(captured["embedding"], np.asarray([[2.0, 3.0]], dtype=np.float32))
    assert captured["kwargs"]["sampling_rate"] == 16_000


def test_tabpfn_runner_verifies_frozen_context_and_v2_settings() -> None:
    registry = ReferenceProfileRegistry.from_file(REFERENCE_CONFIG)
    captured: dict[str, object] = {}

    class Model:
        classes_ = np.asarray([0, 1])

        def fit(self, frame, target):
            captured["fit_frame"] = frame.copy()
            captured["fit_target"] = target.copy()
            return self

        def predict_proba(self, frame):
            captured["predict_frame"] = frame.copy()
            return np.asarray([[0.8, 0.2]])

    def factory(checkpoint, settings):
        captured["checkpoint"] = checkpoint
        captured["settings"] = settings
        return Model()

    checkpoint = Path(
        os.environ.get(
            "RURALSTROKE_TABPFN_CHECKPOINT",
            str(Path.home() / "AppData" / "Roaming" / "tabpfn" / "tabpfn-v2-classifier-finetuned-zk73skhh.ckpt"),
        )
    )
    runner = TabPFNRunner(
        registry,
        checkpoint_path=checkpoint,
        model_factory=factory,
    )
    frame = pd.DataFrame(
        [[68, 1, 0, 130.0, 25.0, "Male", "Yes", "Private", "Urban", "never smoked"]],
        columns=[
            "age", "hypertension", "heart_disease", "avg_glucose_level", "bmi",
            "gender", "ever_married", "work_type", "Residence_type", "smoking_status",
        ],
    )
    probabilities, classes = runner.run(frame)

    assert classes == (0, 1)
    assert probabilities[0, 1] == pytest.approx(0.2)
    assert captured["settings"] == {
        "model_version": "ModelVersion.V2",
        "categorical_feature_indices": [5, 6, 7, 8, 9],
        "ignore_pretraining_limits": True,
    }
    assert list(captured["fit_frame"].columns) == [
        "age", "hypertension", "heart_disease", "avg_glucose_level", "bmi",
        "gender", "ever_married", "work_type", "Residence_type", "smoking_status",
    ]
