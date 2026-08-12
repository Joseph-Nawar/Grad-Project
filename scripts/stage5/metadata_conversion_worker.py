"""Attempt complete frozen metadata pipeline conversion in isolated tooling."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType, StringTensorType


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        model = joblib.load(args.source)
        numeric = {"age", "hypertension", "heart_disease", "avg_glucose_level", "bmi"}
        columns = list(model.feature_names_in_)
        initial_types = [(name, FloatTensorType([None, 1]) if name in numeric else StringTensorType([None, 1])) for name in columns]
        converted = convert_sklearn(model, initial_types=initial_types, target_opset=17, options={id(model): {"zipmap": False}})
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes(converted.SerializeToString())
        print(json.dumps({"status": "SUCCEEDED", "converter_name": "skl2onnx.convert_sklearn", "converter_version": "1.18.0", "runtime_version": "1.20.1", "conversion_options": {"opset": 17, "zipmap": False, "complete_pipeline": True}}))
        return 0
    except Exception as exc:
        print(json.dumps({"status": "FAILED", "failure_reason": f"{type(exc).__name__}: {exc}"}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
