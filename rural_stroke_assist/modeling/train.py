from pathlib import Path

import typer

from rural_stroke_assist.config import MODELS_DIR, PROCESSED_DATA_DIR

app = typer.Typer(help="Training entrypoints for RuralStroke-Assist.")


def train_placeholder(
    features_path: Path,
    labels_path: Path,
    model_path: Path,
) -> str:
    """Return an honest status message for the unimplemented training pipeline."""
    return (
        "Training is not implemented yet. "
        f"Expected features at '{features_path}', labels at '{labels_path}', "
        f"and planned model output at '{model_path}'."
    )


@app.command()
def main(
    features_path: Path = PROCESSED_DATA_DIR / "features.csv",
    labels_path: Path = PROCESSED_DATA_DIR / "labels.csv",
    model_path: Path = MODELS_DIR / "model.pkl",
) -> None:
    """Explain the intended training inputs without pretending a model exists."""
    typer.echo(train_placeholder(features_path, labels_path, model_path))


if __name__ == "__main__":
    app()
