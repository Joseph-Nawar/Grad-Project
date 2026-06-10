from pathlib import Path

import pandas as pd
from PIL import Image


FER2013_LABELS = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "sad",
    "surprise",
]

FER2013_LABEL_MAPPING = {
    label: index for index, label in enumerate(FER2013_LABELS)
}

FER2013_INVERSE_LABEL_MAPPING = {
    index: label for label, index in FER2013_LABEL_MAPPING.items()
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def list_fer2013_image_files(directory: Path) -> list[Path]:
    """Return all image files under the FER2013 directory."""
    return [
        path
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]


def infer_fer2013_split_from_path(image_path: Path) -> str:
    """
    Infer split from FER2013 folder structure.

    Expected Kaggle folder structure:
    fer2013/
      train/
        angry/
        happy/
        ...
      test/
        angry/
        happy/
        ...
    """
    parts = [part.lower() for part in image_path.parts]

    if "train" in parts:
        return "train"

    if "test" in parts:
        return "test"

    return "unknown"


def infer_fer2013_label_from_path(image_path: Path) -> str:
    """Infer emotion label from parent folder name."""
    parent = image_path.parent.name.lower()

    if parent in FER2013_LABEL_MAPPING:
        return parent

    return "unknown"


def is_image_readable(image_path: Path) -> bool:
    """Check whether an image can be opened."""
    try:
        with Image.open(image_path) as image:
            image.verify()
        return True
    except Exception:
        return False


def get_image_size(image_path: Path) -> tuple[int | None, int | None]:
    """Return image width and height."""
    try:
        with Image.open(image_path) as image:
            return image.size
    except Exception:
        return None, None


def build_fer2013_manifest(data_dir: Path) -> pd.DataFrame:
    """
    Build a FER2013 image manifest.

    Columns:
    - path
    - file_name
    - class_label
    - label_encoded
    - split
    - extension
    - readable
    - width
    - height
    """
    image_paths = list_fer2013_image_files(data_dir)

    records = []

    for path in image_paths:
        label = infer_fer2013_label_from_path(path)
        split = infer_fer2013_split_from_path(path)
        readable = is_image_readable(path)
        width, height = get_image_size(path) if readable else (None, None)

        records.append(
            {
                "path": str(path),
                "file_name": path.name,
                "class_label": label,
                "label_encoded": FER2013_LABEL_MAPPING.get(label),
                "split": split,
                "extension": path.suffix.lower(),
                "readable": readable,
                "width": width,
                "height": height,
            }
        )

    return pd.DataFrame(records)


def validate_fer2013_manifest(manifest: pd.DataFrame) -> dict:
    """
    Return basic validation checks for a FER2013 manifest.
    """
    return {
        "total_images": len(manifest),
        "readable_images": int(manifest["readable"].sum()),
        "unknown_labels": int((manifest["class_label"] == "unknown").sum()),
        "unknown_splits": int((manifest["split"] == "unknown").sum()),
        "num_classes": int(manifest["class_label"].nunique()),
        "splits": sorted(manifest["split"].unique().tolist()),
    }