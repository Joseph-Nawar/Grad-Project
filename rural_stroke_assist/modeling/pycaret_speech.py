from __future__ import annotations

import pandas as pd


NON_FEATURE_COLUMNS = {
    "path",
    "file_name",
    "label",
    "label_encoded",
    "speaker_id",
    "torgo_group",
    "split",
}


def prepare_pycaret_speech_dataframe(
    speech_features_df: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Prepare speech feature dataframe for PyCaret.

    Removes non-feature metadata columns but keeps label_encoded as target.
    """
    feature_columns = [
        column
        for column in speech_features_df.columns
        if column not in NON_FEATURE_COLUMNS
    ]

    model_df = speech_features_df[
        feature_columns + ["label_encoded"]
    ].copy()

    return model_df, feature_columns


def split_pycaret_speech_data(
    speech_features_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Return train, validation, and test splits for PyCaret benchmarking.

    The split column is assumed to already be speaker-safe.
    """
    train_df = speech_features_df[
        speech_features_df["split"] == "train"
    ].copy()

    val_df = speech_features_df[
        speech_features_df["split"] == "val"
    ].copy()

    test_df = speech_features_df[
        speech_features_df["split"] == "test"
    ].copy()

    return train_df, val_df, test_df


def build_pycaret_train_data(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Combine train and validation data for PyCaret model comparison.

    Final evaluation is still done on the held-out speaker-safe test set.
    """
    return pd.concat([train_df, val_df], axis=0).reset_index(drop=True)