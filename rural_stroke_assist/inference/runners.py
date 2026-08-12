"""Model-execution runners used beneath the frozen modality adapters."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pandas as pd

from rural_stroke_assist.inference.exceptions import ArtifactConfigurationError


class FaceRunner(Protocol):
    def run(self, tensor: np.ndarray) -> np.ndarray: ...


class TabularRunner(Protocol):
    def run(self, frame: pd.DataFrame) -> tuple[np.ndarray, tuple[Any, ...]]: ...


@dataclass
class OriginalFaceRunner:
    path: Path
    model_loader: Any | None = None
    _model: Any | None = field(init=False, default=None, repr=False)

    def run(self, tensor: np.ndarray) -> np.ndarray:
        try:
            loader = self.model_loader
            if self._model is None and loader is None:
                import tensorflow as tf

                self._model = tf.keras.models.load_model(self.path)
            elif self._model is None:
                self._model = loader(self.path)
            return np.asarray(self._model.predict(tensor, verbose=0))
        except Exception as exc:
            raise ArtifactConfigurationError(f"Unable to run original face model: {self.path}") from exc


@dataclass
class OriginalTabularRunner:
    path: Path
    model_loader: Any | None = None
    _model: Any | None = field(init=False, default=None, repr=False)

    def run(self, frame: pd.DataFrame) -> tuple[np.ndarray, tuple[Any, ...]]:
        try:
            loader = self.model_loader
            if self._model is None and loader is None:
                import joblib

                self._model = joblib.load(self.path)
            elif self._model is None:
                self._model = loader(self.path)
            return np.asarray(self._model.predict_proba(frame), dtype=float), tuple(self._model.classes_)
        except Exception as exc:
            raise ArtifactConfigurationError(
                f"Unable to run original tabular model: {self.path}"
            ) from exc


@dataclass
class LiteRTRunner:
    path: Path
    interpreter_factory: Any | None = None
    _interpreter: Any | None = field(init=False, default=None, repr=False)

    def run(self, tensor: np.ndarray) -> np.ndarray:
        try:
            if self._interpreter is None:
                if self.interpreter_factory is not None:
                    self._interpreter = self.interpreter_factory(self.path)
                else:
                    try:
                        from ai_edge_litert.interpreter import Interpreter
                    except ImportError:
                        try:
                            from tflite_runtime.interpreter import Interpreter
                        except ImportError:
                            import tensorflow as tf

                            Interpreter = tf.lite.Interpreter
                    self._interpreter = Interpreter(model_path=str(self.path))
                self._interpreter.allocate_tensors()
            input_detail = self._interpreter.get_input_details()[0]
            output_detail = self._interpreter.get_output_details()[0]
            self._interpreter.set_tensor(input_detail["index"], tensor)
            self._interpreter.invoke()
            return np.asarray(self._interpreter.get_tensor(output_detail["index"]))
        except Exception as exc:
            raise ArtifactConfigurationError(f"Unable to run LiteRT artifact: {self.path}") from exc


@dataclass
class ONNXRunner:
    path: Path
    session_loader: Any | None = None
    _session: Any | None = field(init=False, default=None, repr=False)

    def _load(self) -> Any:
        if self._session is not None:
            return self._session
        try:
            if self.session_loader is not None:
                self._session = self.session_loader(self.path)
            else:
                import onnxruntime as ort

                self._session = ort.InferenceSession(str(self.path), providers=["CPUExecutionProvider"])
            return self._session
        except Exception as exc:
            raise ArtifactConfigurationError(f"Unable to load ONNX artifact: {self.path}") from exc

    def run(self, value: np.ndarray | pd.DataFrame):
        try:
            session = self._load()
            input_name = session.get_inputs()[0].name
            array = value if isinstance(value, np.ndarray) else value.to_numpy(dtype=np.float32)
            outputs = session.run(None, {input_name: np.asarray(array, dtype=np.float32)})
            if isinstance(value, np.ndarray):
                return np.asarray(outputs[0])
            if len(outputs) == 1:
                probabilities = np.asarray(outputs[0], dtype=float)
            else:
                probabilities = np.asarray(outputs[-1], dtype=float)
                if probabilities.dtype == object:
                    probabilities = np.asarray(
                        [[float(item.get("0", 0.0)), float(item.get("1", 0.0))] for item in outputs[-1]],
                        dtype=float,
                    )
            if probabilities.ndim == 1:
                probabilities = probabilities.reshape(-1, 1)
            return probabilities, tuple(range(probabilities.shape[1]))
        except Exception as exc:
            raise ArtifactConfigurationError(f"Unable to run ONNX artifact: {self.path}") from exc
