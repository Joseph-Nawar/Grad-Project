from __future__ import annotations

from pathlib import Path

from scripts.inspect_stage3_runtime import build_runtime_inventory


def test_runtime_inventory_excludes_development_inputs_and_keeps_canonical_models(tmp_path: Path) -> None:
    (tmp_path / "rural_stroke_assist").mkdir()
    (tmp_path / "rural_stroke_assist" / "server.py").write_text("source", encoding="utf-8")
    (tmp_path / "models" / "metadata").mkdir(parents=True)
    (tmp_path / "models" / "metadata" / "model.joblib").write_bytes(b"model")
    (tmp_path / "data" / "raw").mkdir(parents=True)
    (tmp_path / "data" / "raw" / "sample.csv").write_text("secret", encoding="utf-8")
    (tmp_path / "notebooks").mkdir()
    (tmp_path / "notebooks" / "analysis.ipynb").write_text("{}", encoding="utf-8")
    (tmp_path / "reports").mkdir()
    (tmp_path / "reports" / "old.json").write_text("{}", encoding="utf-8")
    (tmp_path / ".env").write_text("SECRET=value", encoding="utf-8")

    inventory = build_runtime_inventory(tmp_path)

    assert "rural_stroke_assist/server.py" in inventory["source"]
    assert "models/metadata/model.joblib" in inventory["models"]
    assert inventory["excluded"] == {
        "data/raw/sample.csv": "dataset",
        "notebooks/analysis.ipynb": "notebook",
        "reports/old.json": "report",
        ".env": "secret",
    }


def test_runtime_inventory_is_json_serializable_and_sorted(tmp_path: Path) -> None:
    (tmp_path / "models").mkdir()
    (tmp_path / "models" / "z.bin").write_bytes(b"z")
    (tmp_path / "models" / "a.bin").write_bytes(b"a")

    inventory = build_runtime_inventory(tmp_path)

    assert inventory["models"] == ["models/a.bin", "models/z.bin"]
    assert inventory["source"] == []
