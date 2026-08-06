# Speech model card

## Intended use

Research-only dysarthria proxy evidence from the canonical speech branch.

## Out of scope

Stroke diagnosis, clinical speech assessment, language-independent use, or interpretation as a stroke probability.

## Input and preprocessing

Mono audio is resampled to 16 kHz and limited to five seconds. The accepted feature contract contains 37 ordered features: MFCC summaries, duration, RMS, zero-crossing rate, spectral centroid, bandwidth, and rolloff summaries.

## Score semantics

The positive class is `dysarthric`; the output is `dysarthria_proxy_evidence`, not stroke probability.

## Data and results

The canonical manifest is [`data/processed/speech_split_manifest.csv`](../../data/processed/speech_split_manifest.csv), with speaker-held-out test evaluation over 2,849 recordings. Direct ROC-AUC is `0.9426279894`. The runtime adapter accepted 2,744 recordings (`96.31%`) with accepted-subset ROC-AUC `0.9512442545`.

The direct/adapter distinction and reconciliation are recorded in [`speech.json`](../../reports/evaluation/phase4/final_complete/speech.json) and the [Phase 4 report](../../reports/evaluation/phase4/final_complete/PHASE_4_EVALUATION_REPORT.md).

## Artifact and runtime

- Artifact: [`model.pkl`](../../models/experiments/speech/trial_001_mfcc_random_forest/model.pkl)
- SHA256: `1BDED0B3850EE71E2B775E7B8E3BF80F708EE2FAEE1F74CA64D8102BAE01C925`
- Size: 29,499,167 bytes

## Limitations

TORGO dysarthria is not stroke-specific. The branch was developed with English recordings and has not been validated for Arabic or other languages. The artifact was serialized with scikit-learn 1.4.2 and loaded under 1.5.2, producing a compatibility warning. Results are non-diagnostic proxy evidence.
