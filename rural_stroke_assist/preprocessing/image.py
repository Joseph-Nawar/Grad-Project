from pathlib import Path
from typing import Iterable

from PIL import Image


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def list_image_files(directory: Path) -> list[Path]:
    """Return all image files under a directory."""
    return [
        path
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]


def get_image_size(image_path: Path) -> tuple[int, int]:
    """Return image width and height."""
    with Image.open(image_path) as image:
        return image.size


def is_image_readable(image_path: Path) -> bool:
    """Check whether an image can be opened successfully."""
    try:
        with Image.open(image_path) as image:
            image.verify()
        return True
    except Exception:
        return False


def infer_class_from_path(image_path: Path) -> str:
    """
    Infer class label from parent folder name.

    This assumes a folder structure like:
    dataset/class_name/image.jpg
    """
    return image_path.parent.name