from pathlib import Path

import tensorflow as tf


def build_early_stopping_callback(
    patience: int = 3,
    monitor: str = "val_loss",
) -> tf.keras.callbacks.EarlyStopping:
    """
    Stop training when validation performance stops improving.

    restore_best_weights=True ensures the final model uses the best epoch,
    not just the last epoch.
    """
    return tf.keras.callbacks.EarlyStopping(
        monitor=monitor,
        patience=patience,
        restore_best_weights=True,
    )


def build_csv_logger_callback(
    output_path: Path,
) -> tf.keras.callbacks.CSVLogger:
    """
    Save epoch-level training history to CSV.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    return tf.keras.callbacks.CSVLogger(
        filename=str(output_path),
        append=False,
    )