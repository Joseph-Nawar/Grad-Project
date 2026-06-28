# Metadata Feature Contract

## Purpose

This document defines the metadata features used for the MVP metadata model.

The current dataset is the Kaggle Stroke Prediction Dataset. It is used as a contextual risk-factor dataset, not as an acute stroke triage dataset.

## Target

- `stroke`

Binary label:
- `0`: no stroke
- `1`: stroke

## Excluded Columns

- `id`: identifier, not a predictive feature.
- `stroke`: target column.

## Numeric Features

- `age`
- `hypertension`
- `heart_disease`
- `avg_glucose_level`
- `bmi`

## Categorical Features

- `gender`
- `ever_married`
- `work_type`
- `Residence_type`
- `smoking_status`

## Preprocessing Rules

Numeric:
- impute missing values using median
- scale for linear models

Categorical:
- impute missing values using most frequent value
- one-hot encode categories
- handle unknown categories at inference time

## Known Dataset Limitations

- Severe class imbalance: positive stroke class is approximately 4.87%.
- `bmi` has missing values.
- `smoking_status` includes `Unknown`.
- Dataset is not acute FAST-style triage data.
- This model estimates contextual stroke-risk evidence, not definitive diagnosis.