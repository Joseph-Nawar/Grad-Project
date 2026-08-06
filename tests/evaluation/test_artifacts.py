from pathlib import Path
import pytest

from rural_stroke_assist.evaluation.artifact_writer import prepare_output


def test_output_writer_refuses_nonempty_directory(tmp_path: Path):
    target = tmp_path / "run"
    target.mkdir(); (target / "existing").write_text("x")
    with pytest.raises(FileExistsError):
        prepare_output(target)
