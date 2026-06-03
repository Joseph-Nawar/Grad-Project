from pathlib import Path

import typer

from rural_stroke_assist.config import MODELS_DIR, PROCESSED_DATA_DIR

app = typer.Typer(help="Inference entrypoints for RuralStroke-Assist.")


def predict_placeholder(
    features_path: Path,
    model_path: Path,
    predictions_path: Path,
) -> str:
    """Return an honest status message for the unimplemented inference pipeline."""
    return (
        "Inference is not implemented yet. "
        f"Expected features at '{features_path}', model artifact at '{model_path}', "
        f"and planned predictions output at '{predictions_path}'."
    )


@app.command()
def main(
    features_path: Path = PROCESSED_DATA_DIR / "test_features.csv",
    model_path: Path = MODELS_DIR / "model.pkl",
    predictions_path: Path = PROCESSED_DATA_DIR / "test_predictions.csv",
) -> None:
    """Explain the intended inference inputs without pretending predictions exist."""
    typer.echo(predict_placeholder(features_path, model_path, predictions_path))


if __name__ == "__main__":
    app()
