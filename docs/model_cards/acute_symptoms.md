# Acute symptom module card

## Intended use

Deterministic FAST/BE-FAST-style symptom evidence and urgent-rule verification in the research workflow.

## Out of scope

Clinical validation, diagnosis, or replacement of emergency procedures or clinician judgment.

## Input and output

The structured input contains facial droop, arm weakness, speech difficulty, balance/coordination loss, vision disturbance, severe headache, confusion, optional onset minutes, and resolved-symptom state. Output is deterministic evidence in `[0,1]` with LOW/MODERATE/HIGH/URGENT bands and hard-escalation details.

## Evidence

Implementation: [`acute_symptom_module.py`](../../rural_stroke_assist/modules/acute_symptom_module.py). Phase 4 scenario verification: [`symptoms.json`](../../reports/evaluation/phase4/final_complete/symptoms.json).

## Limitations

There is no learned artifact or classifier metric. The rules have not received clinical approval; clinical review status remains pending. Low model evidence must never be used to rule out stroke.
