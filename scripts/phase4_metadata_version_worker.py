"""Run the unchanged metadata artifact in an externally supplied environment."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import joblib
import pandas as pd

parser = argparse.ArgumentParser(); parser.add_argument("--output", required=True); args = parser.parse_args()
frame = pd.read_csv("data/processed/metadata_split_manifest.csv").query("split == 'test'")
columns = ["age", "hypertension", "heart_disease", "avg_glucose_level", "bmi", "gender", "ever_married", "work_type", "Residence_type", "smoking_status"]
model = joblib.load("models/experiments/metadata/mvp_metadata_risk_model.pkl")
scores = model.predict_proba(frame[columns])[:, 1]
Path(args.output).write_text(json.dumps({"scores": [float(value) for value in scores]}), encoding="utf-8")
