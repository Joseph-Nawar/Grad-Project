from pathlib import Path
from collections.abc import Mapping

import matplotlib.pyplot as plt
import pandas as pd


def _coerce_history(history) -> dict:
    """
    Normalize Keras history-like inputs into a plain metrics dictionary.

    Accepts:
    - keras History objects
    - dictionaries like history.history
    - pandas DataFrames with metric columns
    """
    if hasattr(history, "history"):
        history = history.history

    if isinstance(history, pd.DataFrame):
        history = history.to_dict(orient="list")

    if not isinstance(history, Mapping):
        raise TypeError(
            "history must be a Keras History, mapping, or DataFrame"
        )

    return dict(history)


def save_learning_curves(
    history,
    save_path: Path,
):
    """
    Save combined learning curves.

    Parameters
    ----------
    history
        keras History.history dictionary.

    save_path
        Output PNG.
    """

    history = _coerce_history(history)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(history["loss"], label="Train")
    axes[0].plot(history["val_loss"], label="Validation")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    axes[1].plot(history["accuracy"], label="Train")
    axes[1].plot(history["val_accuracy"], label="Validation")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()

    plt.tight_layout()

    save_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    plt.savefig(save_path)

    plt.close(fig)
