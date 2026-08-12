"""Local collector workflow that never depends on central connectivity."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from uuid import uuid4

from pydantic import ValidationError

from rural_stroke_assist.assessment.contracts import AssessmentResult
from rural_stroke_assist.assessment.service import AssessmentService
from rural_stroke_assist.capture.schemas import AssessmentInput
from rural_stroke_assist.cases.attachment_store import AttachmentStore
from rural_stroke_assist.cases.contracts import AttachmentKind
from rural_stroke_assist.cases.serialization import assessment_result_to_dict
from rural_stroke_assist.offline.contracts import AssessmentEnvelope, AttachmentRef
from rural_stroke_assist.offline.store import LocalCase, SQLiteOfflineStore


def _runtime_provenance(assessment_service: AssessmentService) -> dict[str, object]:
    logical = {
        "face": "canonical-face-v1",
        "speech": "canonical-speech-v1",
        "metadata_context": "canonical-metadata-v1",
        "acute_symptoms": "canonical-acute-v1",
    }
    if getattr(assessment_service, "runtime_profile", "original") == "optimized":
        registry_path = Path(__file__).resolve().parents[2] / "config" / "edge_runtime_registry.json"
        registry = json.loads(registry_path.read_text(encoding="utf-8"))
        runtime = {
            modality: {
                "logical_model_id": entry["logical_model_id"],
                "artifact": entry["artifact"],
                "artifact_sha256": entry["artifact_sha256"],
                "backend": entry["backend"],
                "runtime": entry["runtime"],
                "decision": entry["decision"],
            }
            for modality, entry in registry["modalities"].items()
        }
    else:
        baseline = __import__("rural_stroke_assist.inference.registry", fromlist=["load_baseline_registry"]).load_baseline_registry()
        runtime = {
            modality: {
                "logical_model_id": f"canonical-{modality}-v1",
                "artifact": baseline.component(modality).path,
                "artifact_sha256": baseline.data["components"][modality]["sha256"],
                "backend": "TensorFlow/Keras" if modality == "face" else "scikit-learn",
                "runtime": "tensorflow" if modality == "face" else "scikit-learn",
                "decision": "ORIGINAL_RETAINED",
            }
            for modality in ("face", "speech", "metadata_context")
        }
    return {"profile": getattr(assessment_service, "runtime_profile", "original"), "logical_model_ids": logical, "runtime_artifacts": runtime}


class OfflineWorkflow:
    def __init__(self, *, store: SQLiteOfflineStore, attachment_store: AttachmentStore, assessment_service: AssessmentService) -> None:
        self.store = store
        self.attachment_store = attachment_store
        self.assessment_service = assessment_service

    def initialize(self) -> None:
        self.store.initialize()

    def create_draft(self, *, collector_identity: str, facility: str, assessment_input: AssessmentInput, patient_code: str | None = None, face_bytes: bytes | None = None, audio_bytes: bytes | None = None, face_filename: str | None = None, audio_filename: str | None = None, face_media_type: str | None = None, audio_media_type: str | None = None) -> LocalCase:
        if not collector_identity.strip() or not facility.strip():
            raise ValueError("Collector identity and facility are required.")
        case_id = str(uuid4())
        payload = assessment_input.model_dump(mode="json", exclude={"face_image_path", "speech_audio_path", "face_video_path"})
        self.store.create_case(case_id, facility, patient_code, payload, collector_identity=collector_identity)
        if face_bytes:
            reference = self.attachment_store.save(case_id, AttachmentKind.FACE, face_bytes, filename=face_filename or "capture.jpg", media_type=face_media_type)
            self.store.save_attachment(case_id=case_id, attachment_id=str(uuid4()), kind="face", media_type=reference.media_type, size_bytes=reference.size_bytes, sha256=hashlib.sha256(face_bytes).hexdigest(), relative_path=reference.relative_path)
        if audio_bytes:
            reference = self.attachment_store.save(case_id, AttachmentKind.AUDIO, audio_bytes, filename=audio_filename or "recording.wav", media_type=audio_media_type)
            self.store.save_attachment(case_id=case_id, attachment_id=str(uuid4()), kind="audio", media_type=reference.media_type, size_bytes=reference.size_bytes, sha256=hashlib.sha256(audio_bytes).hexdigest(), relative_path=reference.relative_path)
        return self.store.get_case(case_id)

    def assess(self, case_id: str) -> LocalCase:
        case = self.store.get_case(case_id)
        payload = dict(case.assessment_input)
        refs: list[AttachmentRef] = []
        for attachment in self.store.list_attachments(case_id):
            path = self.attachment_store.resolve(type("Reference", (), {"relative_path": attachment.relative_path})())
            if attachment.kind == "face":
                payload["face_image_path"] = path
            elif attachment.kind == "audio":
                payload["speech_audio_path"] = path
            refs.append(AttachmentRef(attachment_id=attachment.attachment_id, kind=attachment.kind, media_type=attachment.media_type, size_bytes=attachment.size_bytes, sha256=attachment.sha256))
        try:
            service_input = AssessmentInput.model_validate(payload)
        except ValidationError as exc:
            raise ValueError("Stored offline assessment input is invalid.") from exc
        result = self.assessment_service.assess(service_input)
        if not isinstance(result, AssessmentResult):
            raise TypeError("Assessment service returned an invalid result.")
        envelope = AssessmentEnvelope(
            case_id=case_id,
            assessment_id=str(uuid4()),
            request={**case.assessment_input, "attachments": [ref.model_dump(mode="json") for ref in refs]},
            result=_portable_result(assessment_result_to_dict(result)),
            provenance={"schema_version": 1, "model_ids": {"face": "canonical-face-v1", "speech": "canonical-speech-v1", "metadata_context": "canonical-metadata-v1", "acute_symptoms": "canonical-acute-v1"}, "fusion_id": "canonical-fusion-v1", **_runtime_provenance(self.assessment_service)},
        )
        return self.store.save_assessment(case_id, envelope)

    def queue(self, case_id: str):
        return self.store.queue_case(case_id)

    def clone_as_draft(self, case_id: str) -> LocalCase:
        original = self.store.get_case(case_id)
        new_case_id = str(uuid4())
        self.store.create_case(new_case_id, original.facility, original.patient_code, original.assessment_input, collector_identity=original.collector_identity)
        for attachment in self.store.list_attachments(case_id):
            source = self.attachment_store.resolve(type("Reference", (), {"relative_path": attachment.relative_path})())
            saved = self.attachment_store.save(new_case_id, AttachmentKind(attachment.kind), source.read_bytes(), filename=Path(attachment.relative_path).name, media_type=attachment.media_type)
            self.store.save_attachment(case_id=new_case_id, attachment_id=str(uuid4()), kind=attachment.kind, media_type=attachment.media_type, size_bytes=saved.size_bytes, sha256=attachment.sha256, relative_path=saved.relative_path)
        return self.store.get_case(new_case_id)


def _portable_result(value: object, *, key: str | None = None) -> object:
    if key == "provenance":
        if isinstance(value, list):
            return ["canonical-runtime-v1"]
        return "canonical-runtime-v1"
    if isinstance(value, dict):
        return {str(name): _portable_result(item, key=str(name)) for name, item in value.items()}
    if isinstance(value, list):
        return [_portable_result(item) for item in value]
    return value
