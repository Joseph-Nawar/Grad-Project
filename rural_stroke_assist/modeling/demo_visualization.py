from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image


def save_face_prediction_demo_figure(
    image_path: str | Path,
    prediction: dict,
    output_path: str | Path,
) -> None:
    """
    Save a report-ready face inference demo figure.

    The figure shows:
    - input image
    - predicted class
    - stroke probability
    - model confidence
    """
    image_path = Path(image_path)
    output_path = Path(output_path)

    image = Image.open(image_path).convert("RGB")

    predicted_label = prediction["predicted_label"]
    stroke_probability = prediction["stroke_probability"]
    confidence = prediction["confidence"]
    threshold = prediction["threshold"]

    fig, ax = plt.subplots(figsize=(7, 8))

    ax.imshow(image)
    ax.axis("off")

    title = (
        f"Face Model Inference Demo\n\n"
        f"Predicted class: {predicted_label}\n"
        f"Stroke probability: {stroke_probability:.3f}\n"
        f"Confidence: {confidence:.3f}\n"
        f"Decision threshold: {threshold:.2f}"
    )

    ax.set_title(title, fontsize=13)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()