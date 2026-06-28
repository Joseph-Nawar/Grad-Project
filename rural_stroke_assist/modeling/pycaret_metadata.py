import pandas as pd

from rural_stroke_assist.preprocessing.metadata import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
)


def split_metadata_pycaret_data(
    metadata_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return train, validation, and test splits."""
    train_df = metadata_df[metadata_df["split"] == "train"].copy()
    val_df = metadata_df[metadata_df["split"] == "val"].copy()
    test_df = metadata_df[metadata_df["split"] == "test"].copy()

    return train_df, val_df, test_df


def prepare_pycaret_metadata_dataframe(
    metadata_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Prepare metadata dataframe for PyCaret.

    Keeps approved feature columns and target.
    """
    return metadata_df[FEATURE_COLUMNS + [TARGET_COLUMN]].copy()


def build_pycaret_metadata_train_data(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Combine train and validation for PyCaret benchmarking.

    Final evaluation remains on held-out test split.
    """
    return pd.concat([train_df, val_df], axis=0).reset_index(drop=True)