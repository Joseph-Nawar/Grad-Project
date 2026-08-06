"""Claim-to-artifact index with explicit unsupported-claim handling."""
from __future__ import annotations
from .contracts import ClaimEvidence, ClaimStatus
from .artifact_writer import write_csv


def default_claims() -> tuple[ClaimEvidence, ...]:
    return (
        ClaimEvidence("Canonical modality metrics are reproducible on held-out proxy partitions", ClaimStatus.SUPPORTED, ("face.json", "speech.json", "metadata_context.json")),
        ClaimEvidence("Face runtime adapter coverage is materially lower than direct model coverage", ClaimStatus.SUPPORTED, ("face_adapter_coverage.json", "face_rejection_by_reason.csv")),
        ClaimEvidence("Speech accepted-subset results are not full held-out coverage results", ClaimStatus.SUPPORTED, ("speech_per_speaker.csv", "speech_speaker_summary.json")),
        ClaimEvidence("Acute symptom rules reach documented deterministic scenarios", ClaimStatus.SUPPORTED, ("symptoms.json",)),
        ClaimEvidence("The fusion implementation is deterministic under missing-modality scenarios", ClaimStatus.SUPPORTED, ("missing_modality_matrix.csv", "ablations.json")),
        ClaimEvidence("Fusion improves clinical stroke performance", ClaimStatus.UNSUPPORTED, (), "No paired multimodal clinical dataset is available."),
        ClaimEvidence("The system has end-to-end stroke diagnostic accuracy", ClaimStatus.UNSUPPORTED, (), "The repository contains proxy branch datasets, not paired clinical outcomes."),
        ClaimEvidence("Clinical safety wording is approved", ClaimStatus.PENDING_HUMAN_REVIEW, (), "Requires a real clinical reviewer response."),
    )


def write_claim_index(path, claims: tuple[ClaimEvidence, ...] | None = None) -> None:
    write_csv(path, ({"claim": c.claim, "status": c.status.value, "evidence": "; ".join(c.evidence), "rationale": c.rationale} for c in (claims or default_claims())), ["claim", "status", "evidence", "rationale"])
