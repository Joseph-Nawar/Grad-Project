import subprocess
from pathlib import Path

from rural_stroke_assist.config import (
    FACE_DATA_DIR,
    METADATA_DATA_DIR,
    FACE_KAGGLE_DATASET,
    METADATA_KAGGLE_DATASET,
)


def ensure_directory(path: Path) -> None:
    """Create a directory if it does not already exist."""
    path.mkdir(parents=True, exist_ok=True)


def download_kaggle_dataset(dataset_slug: str, output_dir: Path) -> None:
    """
    Download and unzip a Kaggle dataset into the target output directory.

    Parameters
    ----------
    dataset_slug:
        Kaggle dataset identifier, e.g. 'username/dataset-name'.
    output_dir:
        Local directory where the dataset should be downloaded.
    """
    ensure_directory(output_dir)

    command = [
        "kaggle",
        "datasets",
        "download",
        "-d",
        dataset_slug,
        "-p",
        str(output_dir),
        "--unzip",
    ]

    print(f"Downloading {dataset_slug} into {output_dir}")
    subprocess.run(command, check=True)


def download_all_datasets() -> None:
    """Download all raw datasets required for the current project milestone."""
    download_kaggle_dataset(FACE_KAGGLE_DATASET, FACE_DATA_DIR)
    download_kaggle_dataset(METADATA_KAGGLE_DATASET, METADATA_DATA_DIR)


if __name__ == "__main__":
    download_all_datasets()