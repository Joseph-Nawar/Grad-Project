"""Pinned project-owned and external-resource identity for the reference profile."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from rural_stroke_assist.inference.exceptions import ArtifactConfigurationError


FEATURE_COLUMNS = [
    "age",
    "hypertension",
    "heart_disease",
    "avg_glucose_level",
    "bmi",
    "gender",
    "ever_married",
    "work_type",
    "Residence_type",
    "smoking_status",
]
NUMERIC_FEATURES = FEATURE_COLUMNS[:5]
CATEGORICAL_FEATURES = FEATURE_COLUMNS[5:]
TARGET_COLUMN = "stroke"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n"
    ).encode("utf-8")


def training_context_fingerprint(frame: pd.DataFrame) -> dict[str, object]:
    """Match the metadata experiment's ordered-row fingerprint exactly."""

    ordered_columns = FEATURE_COLUMNS + [TARGET_COLUMN]
    if "split" in frame.columns:
        ordered_columns.append("split")
    try:
        ordered = frame.loc[:, ordered_columns].copy()
    except KeyError as exc:
        raise ArtifactConfigurationError(
            f"TabPFN context is missing required column(s): {exc}"
        ) from exc
    rows = ordered.astype(object).where(pd.notna(ordered), None).to_dict(orient="records")
    dtype_names = {
        name: (
            "str"
            if name in CATEGORICAL_FEATURES or name == "split"
            else "float64"
            if name in NUMERIC_FEATURES
            else "int64"
        )
        for name in ordered_columns
    }
    payload = {
        "feature_columns": FEATURE_COLUMNS,
        "target_column": TARGET_COLUMN,
        "ordered_columns": ordered_columns,
        "dtypes": dtype_names,
        "rows": rows,
    }
    return {
        "sha256": hashlib.sha256(_stable_json_bytes(payload)).hexdigest(),
        "row_count": int(len(ordered)),
        "feature_columns": FEATURE_COLUMNS,
        "target_column": TARGET_COLUMN,
        "ordered_columns": ordered_columns,
        "dtypes": payload["dtypes"],
    }


@dataclass(frozen=True)
class ReferenceComponent:
    name: str
    path: str
    data: dict[str, Any]

    @property
    def feature_columns(self) -> list[str]:
        return list(self.data.get("feature_columns", []))

    @property
    def thresholds(self) -> dict[str, Any]:
        return dict(self.data.get("thresholds", {}))

    @property
    def provenance(self) -> str:
        return str(self.data.get("provenance", self.path))


class ReferenceProfileRegistry:
    """Small registry facade shared by reference adapters and the factory."""

    def __init__(self, root: Path, data: dict[str, Any]) -> None:
        self.root = root.resolve()
        self.data = data

    @classmethod
    def from_file(cls, path: str | Path | None = None) -> "ReferenceProfileRegistry":
        config_path = (
            Path(path)
            if path is not None
            else Path(__file__).resolve().parents[2] / "config" / "pretrained_reference_registry.json"
        )
        config_path = config_path.resolve()
        try:
            data = json.loads(config_path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise ArtifactConfigurationError(
                f"Unable to read pretrained_reference registry: {config_path}"
            ) from exc
        if data.get("runtime_profile") != "pretrained_reference":
            raise ArtifactConfigurationError("Reference registry is not for pretrained_reference.")
        root_value = data.get("repository_root", data.get("root"))
        root = Path(root_value) if root_value else config_path.parents[1]
        if not root.is_absolute():
            root = config_path.parent / root
        return cls(root, data)

    @property
    def fusion_id(self) -> str:
        return str(self.data.get("fusion_id", "canonical-fusion-v1"))

    def component(self, name: str) -> ReferenceComponent:
        try:
            raw = dict(self.data["modalities"][name])
        except (KeyError, TypeError) as exc:
            raise ArtifactConfigurationError(f"Reference registry has no modality: {name}") from exc
        artifact = str(raw.get("artifact", raw.get("path", "")))
        if not artifact:
            raise ArtifactConfigurationError(f"Reference registry has no artifact for {name}.")
        return ReferenceComponent(name=name, path=artifact, data=raw)

    def path_for(self, name: str) -> Path:
        path = Path(self.component(name).path)
        return path if path.is_absolute() else self.root / path

    def validate_project_artifacts(self) -> None:
        declared_modalities = set(self.data.get("modalities", {}))
        for name in ("face", "speech", "metadata_context"):
            if name not in declared_modalities:
                continue
            component = self.component(name)
            path = self.path_for(name)
            if not path.is_file():
                raise ArtifactConfigurationError(
                    f"Reference {name} artifact is missing: {path}"
                )
            expected = str(component.data.get("artifact_sha256", "")).lower()
            if not expected:
                raise ArtifactConfigurationError(
                    f"Reference {name} artifact hash is not recorded: {path}"
                )
            actual = sha256_file(path).lower()
            if actual != expected:
                raise ArtifactConfigurationError(
                    f"Reference {name} artifact hash mismatch: expected {expected}, got {actual}"
                )

        if "metadata_context" not in declared_modalities:
            return
        metadata = self.component("metadata_context").data
        context_path = self.path_for("metadata_context")
        frame = pd.read_csv(
            context_path,
            dtype={
                **{name: "float64" for name in NUMERIC_FEATURES},
                **{name: "str" for name in CATEGORICAL_FEATURES},
                TARGET_COLUMN: "int64",
            },
        )
        if list(frame.columns) != FEATURE_COLUMNS + [TARGET_COLUMN]:
            raise ArtifactConfigurationError("TabPFN context columns do not match the frozen ten-feature contract.")
        frame["split"] = pd.Series("train", index=frame.index, dtype="str")
        fingerprint = training_context_fingerprint(frame)
        expected_fingerprint = str(metadata.get("training_context_fingerprint", ""))
        if fingerprint["sha256"] != expected_fingerprint:
            raise ArtifactConfigurationError(
                "TabPFN training-context fingerprint mismatch: "
                f"expected {expected_fingerprint}, got {fingerprint['sha256']}"
            )

    def _hf_cache_dir(self) -> Path | None:
        value = os.environ.get("RURALSTROKE_HF_CACHE_DIR") or os.environ.get("HF_HOME")
        return Path(value) if value else None

    def resolve_speech_snapshot(self, *, local_only: bool) -> Path:
        speech = self.component("speech").data
        try:
            from huggingface_hub import snapshot_download

            snapshot = Path(
                snapshot_download(
                    repo_id=str(speech["upstream_model_id"]),
                    revision=str(speech["upstream_revision"]),
                    cache_dir=self._hf_cache_dir(),
                    allow_patterns=list(speech["checkpoint_files"]),
                    local_files_only=local_only,
                )
            )
        except Exception as exc:
            mode = "local" if local_only else "prepared"
            raise ArtifactConfigurationError(
                f"DistilHuBERT {mode} resource is unavailable for revision "
                f"{speech['upstream_revision']}."
            ) from exc
        self.validate_speech_snapshot(Path(snapshot))
        return Path(snapshot)

    def validate_speech_snapshot(self, snapshot: Path) -> None:
        speech = self.component("speech").data
        expected_files = dict(speech.get("checkpoint_file_sha256", {}))
        for name, expected in expected_files.items():
            path = snapshot / name
            if not path.is_file() or sha256_file(path).lower() != str(expected).lower():
                raise ArtifactConfigurationError(
                    f"DistilHuBERT checkpoint identity/hash mismatch for {name}: {path}"
                )

    def resolve_tabpfn_checkpoint(self) -> Path:
        metadata = self.component("metadata_context").data
        filename = str(metadata["checkpoint_filename"])
        explicit = os.environ.get("RURALSTROKE_TABPFN_CHECKPOINT")
        candidates = [Path(explicit)] if explicit else []
        cache_values = [
            os.environ.get("RURALSTROKE_TABPFN_CACHE_DIR"),
            os.environ.get("TABPFN_MODEL_CACHE_DIR"),
        ]
        candidates.extend(Path(value) / filename for value in cache_values if value)
        candidates.extend(
            [
                Path.home() / "AppData" / "Roaming" / "tabpfn" / filename,
                Path.home() / ".cache" / "tabpfn" / filename,
            ]
        )
        path = next((item for item in candidates if item.is_file()), None)
        if path is None:
            raise ArtifactConfigurationError(
                f"TabPFN v2 checkpoint is missing from the prepared local cache: {filename}"
            )
        self.validate_tabpfn_checkpoint(path)
        return path

    def validate_tabpfn_checkpoint(self, path: Path) -> None:
        metadata = self.component("metadata_context").data
        expected = str(metadata["checkpoint_sha256"]).lower()
        actual = sha256_file(path).lower()
        if actual != expected:
            raise ArtifactConfigurationError(
                f"TabPFN v2 checkpoint hash mismatch: expected {expected}, got {actual}"
            )

    def validate_external_resources(self, *, local_only: bool = True) -> dict[str, str]:
        speech_snapshot = self.resolve_speech_snapshot(local_only=local_only)
        tabpfn_checkpoint = self.resolve_tabpfn_checkpoint()
        return {
            "distilhubert_snapshot": str(speech_snapshot),
            "tabpfn_checkpoint": str(tabpfn_checkpoint),
            "resource_mode": "local_only" if local_only else "prepared",
        }

    def provenance(self) -> dict[str, object]:
        return {
            "profile": "pretrained_reference",
            "fusion_id": self.fusion_id,
            "modalities": {
                name: dict(self.component(name).data)
                for name in ("face", "speech", "metadata_context", "acute_symptoms")
            },
        }
