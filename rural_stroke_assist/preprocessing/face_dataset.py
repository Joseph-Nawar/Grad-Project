from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


LABEL_MAPPING = {
    "NonStroke": 0,
    "Stroke": 1,
}

INVERSE_LABEL_MAPPING = {
    0: "NonStroke",
    1: "Stroke",
}


def load_image_as_array(
    image_path: str | Path,
    image_size: tuple[int, int] = (224, 224),
) -> np.ndarray:
    """
    Load an image, convert it to RGB, resize it, and return a NumPy array.

    Output shape:
    (height, width, channels)

    Pixel range:
    0-255
    """
    image = Image.open(image_path).convert("RGB")
    image = image.resize(image_size)

    return np.asarray(image, dtype=np.float32)


def encode_labels(labels: pd.Series) -> np.ndarray:
    """
    Convert string labels to integer labels.

    NonStroke -> 0
    Stroke -> 1
    """
    return labels.map(LABEL_MAPPING).to_numpy(dtype=np.int32)


def load_face_split_arrays(
    split_manifest: pd.DataFrame,
    split: str,
    image_size: tuple[int, int] = (224, 224),
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Load image arrays and labels for one split.

    Returns:
    - X: image tensor
    - y: integer labels
    - split_df: corresponding manifest rows
    """
    split_df = split_manifest[split_manifest["split"] == split].copy()

    images = [
        load_image_as_array(path, image_size=image_size)
        for path in split_df["path"]
    ]

    X = np.stack(images, axis=0)
    y = encode_labels(split_df["class_label"])

    return X, y, split_df.reset_index(drop=True)


def compute_class_weights(y: np.ndarray) -> dict[int, float]:
    """
    Compute inverse-frequency class weights.

    This is preferred over physically duplicating augmented images on disk.
    """
    classes, counts = np.unique(y, return_counts=True)
    total = len(y)

    class_weights = {}

    for cls, count in zip(classes, counts):
        class_weights[int(cls)] = total / (len(classes) * count)

    return class_weights


def compute_class_weights_from_labels(labels: np.ndarray) -> dict[int, float]:
    """
    Compute inverse-frequency class weights from encoded labels.
    """
    classes, counts = np.unique(labels, return_counts=True)
    total = len(labels)

    return {
        int(cls): float(total / (len(classes) * count))
        for cls, count in zip(classes, counts)
    }


def compute_class_weights_from_manifest(df: pd.DataFrame) -> dict[int, float]:
    """
    Compute class weights from a face split manifest.
    """
    encoded_labels = encode_labels(df["class_label"])
    return compute_class_weights_from_labels(encoded_labels)