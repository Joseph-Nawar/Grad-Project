# Data Strategy

## Purpose

This document explains how datasets are used in RuralStroke-Assist and how raw data is transformed into modeling-ready artifacts.

## Data Principles

1. Raw data is immutable.
2. Cleaning decisions must be reproducible.
3. Dataset limitations must be documented.
4. Train/validation/test splits must avoid leakage.
5. Each modality should have its own manifest.

## Face Dataset

The face dataset is used for the face analysis module.

### Known Risks

- Duplicate images
- Cross-class duplicate images
- Class imbalance
- Image size variation
- Possible dataset sourcing limitations

### Cleaning Strategy

Exact duplicate detection is performed using SHA256 file hashes.

- Same-class duplicates are reduced to one representative image.
- Cross-class duplicates are removed entirely.
- The clean manifest is saved under `data/processed`.

## Speech Dataset

The speech dataset is used for the speech abnormality module.

### Known Risks

- Speaker leakage
- Duration variation
- Sample-rate consistency
- Dysarthria is a proxy for stroke-related speech abnormality

### Planned Split Strategy

Speech data should be split by speaker ID, not by audio file.

## Metadata Dataset

The metadata dataset supports contextual risk modeling.

### Known Risks

- Severe class imbalance
- Missing BMI values
- General stroke-risk data, not acute triage data

### Planned Strategy

Use the dataset for baseline tabular modeling and later combine it with structured FAST/NIHSS-inspired symptom metadata.




## Split Strategy

### Face

The cleaned face manifest is split using stratified sampling so that the Stroke/NonStroke ratio is preserved across train, validation, and test sets.

Before splitting, exact duplicates and cross-class duplicates are removed to reduce data leakage.

### Speech

Speech data is split by speaker ID rather than by audio file.

This is critical because random audio-level splitting could place the same speaker in both training and testing sets, causing speaker leakage and inflated performance.

### Metadata

Metadata data is split using stratified sampling because the positive stroke class is rare.

This preserves the severe class imbalance across train, validation, and test sets.