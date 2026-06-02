# RuralStroke-Assist System Architecture

## Objective

RuralStroke-Assist is an offline-capable multimodal AI decision-support system for rural stroke triage.

## Architecture

Input gathering  
→ Quality assessment  
→ Preprocessing  
→ Face module  
→ Speech module  
→ Metadata module  
→ Fusion engine  
→ Report generator

## MVP Inputs

- Face image/video
- Speech audio
- Patient metadata

## MVP Outputs

- Triage level
- Risk score
- Confidence
- Explanation
- Safety warning

## Safety Framing

The system does not diagnose stroke. It provides triage decision support.