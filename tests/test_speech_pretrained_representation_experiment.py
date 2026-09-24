from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
import scripts.speech_pretrained_representation_experiment as speech_experiment
from scripts.speech_pretrained_representation_experiment import (
    _extract_or_load_split,
    calculate_binary_metrics,
    fit_train_only_classifier,
    make_cache_key,
    summarize_by_speaker,
    validate_speaker_partitions,
)
from sklearn.preprocessing import StandardScaler


def test_speaker_partition_validation_rejects_cross_split_speakers() -> None:
    frame = pd.DataFrame(
        {
            "speaker_id": ["control_a", "control_a", "patient_b", "patient_c"],
            "label": ["control", "control", "dysarthric", "dysarthric"],
            "split": ["train", "test", "val", "test"],
        }
    )

    with pytest.raises(ValueError, match="appears in multiple splits"):
        validate_speaker_partitions(frame)


def test_speaker_partition_validation_checks_expected_assignments() -> None:
    frame = pd.DataFrame(
        {
            "speaker_id": ["control_a", "patient_b", "control_c"],
            "label": ["control", "dysarthric", "control"],
            "split": ["train", "val", "test"],
        }
    )

    validate_speaker_partitions(
        frame,
        expected={
            "train": {"control_a"},
            "val": {"patient_b"},
            "test": {"control_c"},
        },
    )

    with pytest.raises(ValueError, match="speaker partition mismatch"):
        validate_speaker_partitions(
            frame,
            expected={
                "train": {"control_c"},
                "val": {"patient_b"},
                "test": {"control_a"},
            },
        )


def test_embedding_cache_key_changes_with_model_revision_sample_and_split() -> None:
    base = {
        "model_id": "candidate/model",
        "model_revision": "commit-a",
        "preprocessing": {"sample_rate": 16000, "pooling": "mean"},
        "sample_sha256": "sample-a",
        "split": "train",
        "split_manifest_sha256": "manifest-a",
    }

    key = make_cache_key(**base)

    assert key == make_cache_key(**base)
    assert key != make_cache_key(**{**base, "model_revision": "commit-b"})
    assert key != make_cache_key(**{**base, "sample_sha256": "sample-b"})
    assert key != make_cache_key(**{**base, "split": "test"})


def test_embedding_cache_reuses_only_matching_sample_and_revision(tmp_path, monkeypatch) -> None:
    audio_path = tmp_path / "recording.wav"
    audio_path.write_bytes(b"sample-a")

    def fake_load_audio(audio_path, target_sample_rate, max_duration_seconds):
        return np.linspace(-0.5, 0.5, 8000, dtype=np.float32), target_sample_rate

    monkeypatch.setattr(speech_experiment, "load_audio", fake_load_audio)
    frame = pd.DataFrame(
        {
            "_sample_id": [7],
            "path": [str(audio_path)],
            "speaker_id": ["speaker_a"],
            "label": ["control"],
            "label_encoded": [0],
            "split": ["train"],
        }
    )

    class FakeEncoder:
        dimension = 2

        def __init__(self):
            self.calls = 0

        def embed(self, signal):
            self.calls += 1
            return np.array([signal.mean(), signal.std()], dtype=np.float32)

    encoder = FakeEncoder()
    kwargs = {
        "encoder": encoder,
        "candidate_id": "distilhubert",
        "split": "train",
        "split_manifest_sha256": "manifest-a",
        "cache_dir": tmp_path / "cache",
    }
    _, first = _extract_or_load_split(frame, revision="commit-a", **kwargs)
    _, repeated = _extract_or_load_split(frame, revision="commit-a", **kwargs)
    _, changed_revision = _extract_or_load_split(frame, revision="commit-b", **kwargs)
    audio_path.write_bytes(b"sample-b")
    _, changed_sample = _extract_or_load_split(frame, revision="commit-a", **kwargs)
    _, changed_split = _extract_or_load_split(frame, revision="commit-a", split="val", **{
        key: value for key, value in kwargs.items() if key != "split"
    })

    assert first["cache_hit"] is False
    assert repeated["cache_hit"] is True
    assert changed_revision["cache_hit"] is False
    assert changed_sample["cache_hit"] is False
    assert changed_split["cache_hit"] is False
    assert encoder.calls == 4


def test_classifier_fit_uses_only_finite_training_embeddings() -> None:
    frame = pd.DataFrame(
        {
            "split": ["train", "train", "val", "test"],
            "label_encoded": [0, 1, 0, 1],
        }
    )
    embeddings = np.array([[0.0, 0.0], [2.0, 2.0], [100.0, 100.0], [-100.0, -100.0]])

    pipeline, fit_summary = fit_train_only_classifier(frame, embeddings)

    scaler = pipeline.named_steps["scaler"]
    assert isinstance(scaler, StandardScaler)
    np.testing.assert_allclose(scaler.mean_, [1.0, 1.0])
    assert fit_summary == {"recording_count": 2, "speaker_count": None}


def test_metrics_and_single_class_speaker_auc_are_reported_safely() -> None:
    labels = np.array([0, 0, 1, 1])
    probabilities = np.array([0.1, 0.8, 0.7, 0.9])
    metrics = calculate_binary_metrics(labels, probabilities)

    assert metrics["confusion_matrix"] == [[1, 1], [0, 2]]
    assert metrics["sensitivity"] == 1.0
    assert metrics["specificity"] == 0.5
    assert metrics["roc_auc"] == 0.75
    assert metrics["pr_auc"] == pytest.approx(0.8333333333333333)

    frame = pd.DataFrame(
        {
            "speaker_id": ["control_a", "control_a", "patient_b", "patient_b"],
            "label_encoded": labels,
        }
    )
    summaries = summarize_by_speaker(frame, probabilities)

    assert {row["speaker_id"] for row in summaries} == {"control_a", "patient_b"}
    assert all(row["roc_auc"] is None for row in summaries)


def test_metric_replay_can_use_estimator_predictions_for_probability_ties() -> None:
    metrics = calculate_binary_metrics(
        [0, 1, 0, 1],
        [0.5, 0.5, 0.2, 0.8],
        predictions=[0, 0, 0, 1],
    )

    assert metrics["confusion_matrix"] == [[2, 0], [1, 1]]
    assert metrics["accuracy"] == 0.75
    assert metrics["sensitivity"] == 0.5


def test_report_derives_historical_accuracy_and_summarizes_speaker_ranges(tmp_path) -> None:
    comparison = pd.DataFrame(
        [
            {"candidate_id": "yamnet", "status": "completed", "validation_roc_auc": 0.7, "validation_pr_auc": 0.6,
             "validation_accuracy": 0.7, "validation_sensitivity": 0.6, "validation_specificity": 0.8,
             "validation_precision": 0.7, "validation_f1": 0.6, "validation_n": 4, "validation_speakers": 2},
            {"candidate_id": "distilhubert", "status": "completed", "validation_roc_auc": 0.8, "validation_pr_auc": 0.7,
             "validation_accuracy": 0.8, "validation_sensitivity": 0.7, "validation_specificity": 0.9,
             "validation_precision": 0.8, "validation_f1": 0.7, "validation_n": 4, "validation_speakers": 2},
            {"candidate_id": "wav2vec2_base", "status": "completed", "validation_roc_auc": 0.6, "validation_pr_auc": 0.5,
             "validation_accuracy": 0.6, "validation_sensitivity": 0.5, "validation_specificity": 0.7,
             "validation_precision": 0.6, "validation_f1": 0.5, "validation_n": 4, "validation_speakers": 2},
        ]
    )
    comparison.to_csv(tmp_path / "validation_comparison.csv", index=False)
    (tmp_path / "baseline_validation.json").write_text(
        json.dumps({"roc_auc": 0.5, "pr_auc": 0.4, "accuracy": 0.6, "sensitivity": 0.4, "specificity": 0.8,
                    "precision": 0.5, "f1": 0.4, "confusion_matrix": [[2, 1], [1, 0]],
                    "accepted_sample_count": 4, "speaker_count": 2}),
        encoding="utf-8",
    )
    (tmp_path / "baseline_historical_test.json").write_text(
        json.dumps({"candidate_id": "mfcc_random_forest_baseline", "roc_auc": 0.7, "pr_auc": 0.6,
                    "accuracy": 0.7, "sensitivity": 0.6, "specificity": 0.8, "precision": 0.7, "f1": 0.6,
                    "confusion_matrix": [[2, 1], [1, 1]], "accepted_sample_count": 5, "speaker_count": 2,
                    "phase4_direct_report": {"n": 5, "confusion_matrix": [[2, 1], [0, 2]]}}),
        encoding="utf-8",
    )
    speaker_rows = []
    for candidate_id in ("yamnet", "distilhubert", "wav2vec2_base"):
        speaker_rows.extend(
            [
                {"candidate_id": candidate_id, "speaker_id": "positive_a", "label": "dysarthric",
                 "sensitivity": 0.5, "specificity": np.nan},
                {"candidate_id": candidate_id, "speaker_id": "positive_b", "label": "dysarthric",
                 "sensitivity": 0.7, "specificity": np.nan},
                {"candidate_id": candidate_id, "speaker_id": "control_a", "label": "control",
                 "sensitivity": np.nan, "specificity": 0.75},
                {"candidate_id": candidate_id, "speaker_id": "control_b", "label": "control",
                 "sensitivity": np.nan, "specificity": 0.9},
            ]
        )
    pd.DataFrame(speaker_rows).to_csv(tmp_path / "validation_per_speaker.csv", index=False)
    for candidate_id in ("yamnet", "distilhubert", "wav2vec2_base"):
        (tmp_path / f"validation_{candidate_id}.json").write_text(
            json.dumps({"status": "completed", "resources": {}}), encoding="utf-8"
        )

    speech_experiment._write_report(tmp_path)
    report = (tmp_path / "REPORT.md").read_text(encoding="utf-8")

    assert "accuracy 0.8000" in report
    assert "yamnet: validation sensitivity 0.500 to 0.700, specificity 0.750 to 0.900" in report


def test_selection_cannot_be_reset_after_the_test_pass(tmp_path) -> None:
    (tmp_path / "selection.json").write_text(
        json.dumps({"test_evaluated": True}), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="selection cannot be replaced or reset"):
        speech_experiment.run_selection(output_dir=tmp_path)


def test_test_stage_cannot_be_repeated(tmp_path) -> None:
    (tmp_path / "selection.json").write_text(
        json.dumps({"test_evaluated": True}), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="test stage has already run"):
        speech_experiment.run_test(output_dir=tmp_path)
