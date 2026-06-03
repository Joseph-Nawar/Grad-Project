# Dataset Sources

## Annotated Facial Images for Stroke Classification

Source: Kaggle  
Dataset slug: abdussalamelhanashy/annotated-facial-images-for-stroke-classification

Purpose:
Used as the initial face-image dataset for the face analysis module.

Known limitation:
Useful for prototyping, but not sufficient for clinical validation.

## TORGO Audio Dataset

Source: Kaggle mirror of TORGO  
Kaggle slug: pranaykoppula/torgo-audio  
Official TORGO reference: University of Toronto TORGO Database

Purpose:
Used as the initial public speech dataset for the speech analysis module.

Project relevance:
The dataset contains dysarthric and non-dysarthric speech. Dysarthria is a useful proxy for impaired or slurred speech, which is relevant to the FAST speech-warning component in stroke triage.

Known limitation:
TORGO is not stroke-specific and should not be treated as clinical stroke speech data.

## Stroke Prediction Dataset

Source: Kaggle  
Dataset slug: fedesoriano/stroke-prediction-dataset

Purpose:
Used as a metadata/contextual risk dataset.

Known limitation:
This is a general stroke risk dataset, not an acute prehospital FAST dataset.