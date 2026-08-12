"""Representative full-assessment fused risk-band parity suite."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd


def run_assessment_parity(root: Path) -> dict:
    sys.path.insert(0, str(root))
    from rural_stroke_assist.assessment.fusion_strategy import CanonicalLateFusionStrategy
    from rural_stroke_assist.assessment.service import AssessmentService
    from rural_stroke_assist.capture.acute_symptom_schema import AcuteStrokeSymptoms
    from rural_stroke_assist.capture.schemas import AssessmentInput
    from rural_stroke_assist.inference.face_adapter import FaceAdapter
    from rural_stroke_assist.inference.metadata_adapter import MetadataAdapter, MetadataInput
    from rural_stroke_assist.inference.registry import BaselineRegistry
    from rural_stroke_assist.inference.runners import LiteRTRunner, ONNXRunner, OriginalFaceRunner, OriginalTabularRunner
    from rural_stroke_assist.inference.speech_adapter import SpeechAdapter
    from rural_stroke_assist.inference.symptom_adapter import SymptomAdapter

    registry = BaselineRegistry.from_file(root / "config/baseline_registry.json")
    face = pd.read_csv(registry.manifest_path("face_split")); face_path = Path(face.loc[face["split"] == "test", "path"].iloc[0])
    speech = pd.read_csv(registry.manifest_path("speech_split")); speech_path = Path(speech.loc[speech["split"] == "test", "path"].iloc[0])
    meta = pd.read_csv(registry.manifest_path("metadata_split")); row = meta.loc[meta["split"] == "test"].iloc[0]
    metadata = MetadataInput(age=row.age, hypertension=row.hypertension, heart_disease=row.heart_disease, avg_glucose_level=row.avg_glucose_level, bmi=row.bmi, gender=row.gender, ever_married=row.ever_married, work_type=row.work_type, Residence_type=row.Residence_type, smoking_status=row.smoking_status)
    original = AssessmentService(adapters={"face": FaceAdapter(registry=registry, runner=OriginalFaceRunner(registry.path_for("face"))), "speech": SpeechAdapter(registry=registry, runner=OriginalTabularRunner(registry.path_for("speech"))), "metadata_context": MetadataAdapter(registry=registry), "acute_symptoms": SymptomAdapter(registry=registry)}, fusion_strategy=CanonicalLateFusionStrategy(), runtime_profile="original")
    mixed = AssessmentService(adapters={"face": FaceAdapter(registry=registry, runner=LiteRTRunner(root / ".stage5/candidates/face/face-litert-fp32.tflite")), "speech": SpeechAdapter(registry=registry, runner=ONNXRunner(root / ".stage5/candidates/speech/speech-onnx-random-forest-pipeline.onnx")), "metadata_context": MetadataAdapter(registry=registry), "acute_symptoms": SymptomAdapter(registry=registry)}, fusion_strategy=CanonicalLateFusionStrategy(), runtime_profile="optimized")
    scenarios = {
        "complete_urgent": {"face_image_path": face_path, "speech_audio_path": speech_path, "metadata": metadata, "acute_symptoms": AcuteStrokeSymptoms(face_drooping=True, arm_weakness=True)},
        "complete_nonurgent": {"face_image_path": face_path, "speech_audio_path": speech_path, "metadata": metadata, "acute_symptoms": AcuteStrokeSymptoms()},
        "face_missing": {"speech_audio_path": speech_path, "metadata": metadata, "acute_symptoms": AcuteStrokeSymptoms()},
        "speech_missing": {"face_image_path": face_path, "metadata": metadata, "acute_symptoms": AcuteStrokeSymptoms()},
        "metadata_missing": {"face_image_path": face_path, "speech_audio_path": speech_path, "acute_symptoms": AcuteStrokeSymptoms()},
        "symptoms_only": {"acute_symptoms": AcuteStrokeSymptoms(face_drooping=True)},
        "empty": {},
    }
    results = []
    for name, values in scenarios.items():
        original_result = original.assess(AssessmentInput(session_id=f"parity-original-{name}", **values))
        mixed_result = mixed.assess(AssessmentInput(session_id=f"parity-mixed-{name}", **values))
        original_band = original_result.fusion.risk_band if original_result.fusion else None
        mixed_band = mixed_result.fusion.risk_band if mixed_result.fusion else None
        results.append({"scenario": name, "original_band": original_band, "optimized_band": mixed_band, "agreement": original_band == mixed_band, "original_status": original_result.status.value, "optimized_status": mixed_result.status.value, "original_scores": {key: execution.evidence.score for key, execution in original_result.modality_executions.items()}, "optimized_scores": {key: execution.evidence.score for key, execution in mixed_result.modality_executions.items()}, "optimized_failures": {key: execution.failure.error_type if execution.failure else None for key, execution in mixed_result.modality_executions.items()}})
    agreement = sum(item["agreement"] for item in results) / len(results)
    payload = {"schema_version": 1, "scenario_count": len(results), "risk_band_agreement": agreement, "pass": agreement == 1.0, "results": results}
    output = root / "reports/production/stage5/assessment_parity.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def main() -> int:
    root = Path(__file__).resolve().parents[2]
    payload = run_assessment_parity(root)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
