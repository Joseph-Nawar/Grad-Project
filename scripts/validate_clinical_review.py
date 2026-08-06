from __future__ import annotations
import argparse
from rural_stroke_assist.evaluation.clinical_review import validate_review

parser = argparse.ArgumentParser()
parser.add_argument("input")
parser.add_argument("--output-dir", default="reports/evaluation/phase4/clinical_review")
args = parser.parse_args()
status, log = validate_review(args.input, args.output_dir)
print(status)
print(log)
