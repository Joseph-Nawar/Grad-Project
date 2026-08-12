"""Run one face conversion inside the isolated Stage 5 Python environment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        import tensorflow as tf

        model = tf.keras.models.load_model(args.source)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        if args.candidate.startswith("face-litert"):
            converter = tf.lite.TFLiteConverter.from_keras_model(model)
            options = {"precision": "fp32"}
            if args.candidate.endswith("dynamic-range"):
                converter.optimizations = [tf.lite.Optimize.DEFAULT]
                options = {"quantization": "dynamic-range", "calibration": "none"}
            blob = converter.convert()
            args.output.write_bytes(blob)
            print(json.dumps({
                "status": "SUCCEEDED",
                "converter_name": "tensorflow.lite.TFLiteConverter",
                "converter_version": tf.__version__,
                "runtime_name": "tensorflow-lite",
                "runtime_version": tf.__version__,
                "conversion_options": options,
                "input_signature": "(batch,160,160,3) float32",
                "output_signature": "(batch,1) float32",
            }))
            return 0
        import tf2onnx

        signature = (tf.TensorSpec((None, 160, 160, 3), tf.float32, name="input"),)
        tf2onnx.convert.from_keras(model, input_signature=signature, opset=17, output_path=str(args.output))
        print(json.dumps({
            "status": "SUCCEEDED",
            "converter_name": "tf2onnx.convert.from_keras",
            "converter_version": getattr(tf2onnx, "__version__", "1.16.1"),
            "runtime_name": "onnxruntime",
            "runtime_version": "1.20.1",
            "conversion_options": {"opset": 17, "input_name": "input"},
            "input_signature": "(batch,160,160,3) float32",
            "output_signature": "(batch,1) float32",
        }))
        return 0
    except Exception as exc:
        print(json.dumps({"status": "FAILED", "failure_reason": f"{type(exc).__name__}: {exc}"}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
