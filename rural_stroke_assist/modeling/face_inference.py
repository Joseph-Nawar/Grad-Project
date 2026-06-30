from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image


FACE_LABELS = {
    0: "NonStroke",
    1: "Stroke",
}


def load_face_model(model_path: str | Path) -> tf.keras.Model:
    """
    Load a saved Keras face classification model.
    """
    model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(f"Face model not found: {model_path}")

    return tf.keras.models.load_model(model_path)


def preprocess_face_image(
    image_path: str | Path,
    image_size: tuple[int, int] = (160, 160),
) -> np.ndarray:
    """
    Load and preprocess a single image for face model inference.

    Output shape:
    (1, height, width, 3)

    Pixel range:
    0-1
    """
    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    image = Image.open(image_path).convert("RGB")
    image = image.resize(image_size)

    image_array = np.asarray(image, dtype=np.float32) / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    return image_array


def predict_face_image(
    model: tf.keras.Model,
    image_path: str | Path,
    image_size: tuple[int, int] = (160, 160),
    threshold: float = 0.5,
) -> dict:
    """
    Run inference on one face image.

    Returns:
    - stroke_probability
    - predicted_label_encoded
    - predicted_label
    - confidence
    """
    image_array = preprocess_face_image(
        image_path=image_path,
        image_size=image_size,
    )

    raw_prediction = model.predict(image_array, verbose=0)

    stroke_probability = float(raw_prediction.reshape(-1)[0])

    predicted_label_encoded = int(stroke_probability >= threshold)
    predicted_label = FACE_LABELS[predicted_label_encoded]

    confidence = (
        stroke_probability
        if predicted_label_encoded == 1
        else 1.0 - stroke_probability
    )

    return {
        "image_path": str(image_path),
        "stroke_probability": stroke_probability,
        "predicted_label_encoded": predicted_label_encoded,
        "predicted_label": predicted_label,
        "confidence": confidence,
        "threshold": threshold,
    }