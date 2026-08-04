"""Access to the Phase 0 baseline registry."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RegistryComponent:
    name: str
    path: str
    data: dict[str, Any]

    @property
    def feature_columns(self) -> list[str]:
        return list(self.data.get("feature_columns", []))

    @property
    def thresholds(self) -> dict[str, Any]:
        return dict(self.data.get("thresholds", {}))


class BaselineRegistry:
    def __init__(self, root: Path, data: dict[str, Any]) -> None:
        self.root = root.resolve()
        self.data = data

    @classmethod
    def from_file(cls, path: str | Path) -> "BaselineRegistry":
        registry_path = Path(path).resolve()
        with registry_path.open("r", encoding="utf-8") as stream:
            data = json.load(stream)
        if not isinstance(data, dict) or "components" not in data:
            raise ValueError("Invalid baseline registry: components section is required.")
        return cls(registry_path.parent.parent, data)

    @property
    def fusion(self) -> dict[str, Any]:
        return dict(self.data["fusion"])

    def component(self, name: str) -> RegistryComponent:
        try:
            component = self.data["components"][name]
        except KeyError as exc:
            raise KeyError(f"Unknown baseline component: {name}") from exc
        return RegistryComponent(name=name, path=component["path"], data=dict(component))

    def path_for(self, name: str) -> Path:
        return self.root / self.component(name).path

    def manifest_path(self, name: str) -> Path:
        return self.root / self.data["manifests"][name]["path"]


def load_baseline_registry(path: str | Path | None = None) -> BaselineRegistry:
    if path is None:
        path = Path(__file__).resolve().parents[2] / "config" / "baseline_registry.json"
    return BaselineRegistry.from_file(path)
