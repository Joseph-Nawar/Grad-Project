from pathlib import Path
import json
from rural_stroke_assist.evaluation.gap_closure import metadata_version_comparison


def test_version_comparison_records_structured_skip(tmp_path: Path):
    result = metadata_version_comparison(tmp_path, str(tmp_path / "missing-python"))
    assert result["status"] == "SKIPPED"
    assert json.loads((tmp_path / "metadata_version_comparison.json").read_text())["status"] == "SKIPPED"
