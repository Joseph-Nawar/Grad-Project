from pathlib import Path
import hashlib

import pandas as pd


def compute_file_hash(file_path: Path, chunk_size: int = 8192) -> str:
    """
    Compute a SHA256 hash for a file.

    The hash uniquely represents the file content.
    If two files have the same hash, they are exact duplicates.
    """
    sha256 = hashlib.sha256()

    with file_path.open("rb") as file:
        for chunk in iter(lambda: file.read(chunk_size), b""):
            sha256.update(chunk)

    return sha256.hexdigest()


def add_file_hashes(manifest: pd.DataFrame, path_column: str = "path") -> pd.DataFrame:
    """
    Add a file_hash column to an image manifest.

    Parameters
    ----------
    manifest:
        DataFrame containing image file paths.
    path_column:
        Column containing image paths.
    """
    manifest = manifest.copy()

    manifest["file_hash"] = manifest[path_column].apply(
        lambda path: compute_file_hash(Path(path))
    )

    return manifest


def create_duplicate_report(
    manifest: pd.DataFrame,
    hash_column: str = "file_hash",
    label_column: str = "class_label",
) -> pd.DataFrame:
    """
    Create a duplicate report grouped by file hash.

    Each row represents one unique image hash.

    Important fields:
    - file_count: how many times the same image appears
    - class_count: how many different labels it appears under
    - labels: labels associated with the hash
    - is_duplicate: whether the file appears more than once
    - is_cross_class_duplicate: whether same image appears under multiple labels
    """
    grouped = (
        manifest.groupby(hash_column)
        .agg(
            file_count=(hash_column, "size"),
            class_count=(label_column, "nunique"),
            labels=(label_column, lambda values: sorted(values.unique())),
            paths=("path", lambda values: list(values)),
        )
        .reset_index()
    )

    grouped["is_duplicate"] = grouped["file_count"] > 1
    grouped["is_cross_class_duplicate"] = grouped["class_count"] > 1

    return grouped.sort_values(
        by=["is_cross_class_duplicate", "file_count"],
        ascending=[False, False],
    )


def mark_duplicate_status(
    manifest: pd.DataFrame,
    duplicate_report: pd.DataFrame,
    hash_column: str = "file_hash",
) -> pd.DataFrame:
    """
    Add duplicate flags to the original manifest.
    """
    flags = duplicate_report[
        [
            hash_column,
            "file_count",
            "class_count",
            "is_duplicate",
            "is_cross_class_duplicate",
        ]
    ]

    return manifest.merge(flags, on=hash_column, how="left")


def create_clean_face_manifest(
    manifest: pd.DataFrame,
    hash_column: str = "file_hash",
    label_column: str = "class_label",
) -> pd.DataFrame:
    """
    Create a cleaned face manifest.

    Cleaning strategy:
    1. Remove all cross-class duplicates because the same exact image has conflicting labels.
    2. For remaining duplicate groups, keep only one image per hash.
    3. Preserve class labels for non-conflicting unique images.

    This produces a clean manifest for future modeling.
    """
    manifest = manifest.copy()

    # Remove hashes that appear under multiple labels.
    cross_class_hashes = set(
        manifest.groupby(hash_column)[label_column]
        .nunique()
        .loc[lambda series: series > 1]
        .index
    )

    without_cross_class = manifest[
        ~manifest[hash_column].isin(cross_class_hashes)
    ].copy()

    # Keep one representative image per exact file hash.
    clean_manifest = (
        without_cross_class
        .sort_values(by=["class_label", "path"])
        .drop_duplicates(subset=[hash_column], keep="first")
        .reset_index(drop=True)
    )

    clean_manifest["is_clean"] = True

    return clean_manifest


def summarize_duplicate_audit(
    manifest: pd.DataFrame,
    duplicate_report: pd.DataFrame,
    clean_manifest: pd.DataFrame,
) -> dict:
    """
    Return key duplicate-audit metrics as a dictionary.
    """
    total_files = len(manifest)
    unique_hashes = manifest["file_hash"].nunique()
    duplicate_extra_files = total_files - unique_hashes

    cross_class_duplicate_hashes = duplicate_report[
        duplicate_report["is_cross_class_duplicate"]
    ]["file_hash"].nunique()

    return {
        "total_files": total_files,
        "unique_hashes": unique_hashes,
        "duplicate_extra_files": duplicate_extra_files,
        "duplicate_extra_percentage": round(
            duplicate_extra_files / total_files * 100, 2
        )
        if total_files
        else 0.0,
        "cross_class_duplicate_hashes": cross_class_duplicate_hashes,
        "clean_files": len(clean_manifest),
        "removed_files": total_files - len(clean_manifest),
    }