# Phase 4 clinical safety review status

**Status: PENDING. No signed clinician review or clinical safety approval artifact is present in this repository.**

This status document records the absence of an approval artifact. It is not itself a clinical review, approval, or validation.

## Current safety boundary

- RuralStroke-Triage is a research and engineering prototype, not a diagnostic system and not for use to determine care.
- The modality branches return heterogeneous proxy or structured evidence. They are not calibrated clinical stroke probabilities.
- The visual and speech datasets are proxies. Speech data concern dysarthria and are not stroke-specific. Contextual metadata represents background risk, not an acute presentation.
- Acute symptom safeguards are deterministic implementation rules; their presence does not establish clinical approval or validated escalation behavior.
- There is no paired patient-level multimodal evaluation, prospective study, clinical utility study, or native smartphone validation in the checked-in evidence.

## Before any clinical-use or clinical-performance claim

A qualified clinical reviewer must examine intended use, user-facing language, foreseeable misuse, emergency escalation, failure and abstention states, missing modalities, human-review responsibilities, and the limits of each input dataset. Any future study must separately document ethics approval, consent, data rights, protocol, population, ground truth, and analysis plan.

## Source records

- Project scope and current limitations: `README.md`, sections “Limitations and responsible use” and “Project status”.
- Current evaluation boundaries: `reports/evaluation/phase4/final_complete/PHASE_4_EVALUATION_REPORT.md` and `docs/model_cards/`.
- Dataset uncertainties: `docs/dataset_cards/`.
- Evidence qualification and reproduction limitations: `docs/demo/ruralstroke-evidence-report.md`.
