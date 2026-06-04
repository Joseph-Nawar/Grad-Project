import pandas as pd
from sklearn.model_selection import train_test_split


def add_stratified_split(
    df: pd.DataFrame,
    label_column: str,
    train_size: float = 0.70,
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Add train/val/test split labels using stratified sampling.

    Best for datasets where each row can be split independently,
    such as deduplicated image datasets or tabular datasets.
    """
    if round(train_size + val_size + test_size, 5) != 1.0:
        raise ValueError("train_size + val_size + test_size must equal 1.0")

    df = df.copy()

    train_df, temp_df = train_test_split(
        df,
        train_size=train_size,
        stratify=df[label_column],
        random_state=random_state,
    )

    relative_val_size = val_size / (val_size + test_size)

    val_df, test_df = train_test_split(
        temp_df,
        train_size=relative_val_size,
        stratify=temp_df[label_column],
        random_state=random_state,
    )

    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()

    train_df["split"] = "train"
    val_df["split"] = "val"
    test_df["split"] = "test"

    return (
        pd.concat([train_df, val_df, test_df], axis=0)
        .sort_index()
        .reset_index(drop=True)
    )


def add_group_split(
    df: pd.DataFrame,
    group_column: str,
    label_column: str,
    train_size: float = 0.70,
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Add train/val/test split labels using group-level splitting.

    This is required when rows are not independent.

    Example:
    Speech data must be split by speaker_id, not by audio file,
    to prevent the same speaker appearing in train and test.
    """
    if round(train_size + val_size + test_size, 5) != 1.0:
        raise ValueError("train_size + val_size + test_size must equal 1.0")

    group_labels = (
        df[[group_column, label_column]]
        .drop_duplicates(subset=[group_column])
        .reset_index(drop=True)
    )

    train_groups, temp_groups = train_test_split(
        group_labels,
        train_size=train_size,
        stratify=group_labels[label_column],
        random_state=random_state,
    )

    relative_val_size = val_size / (val_size + test_size)

    val_groups, test_groups = train_test_split(
        temp_groups,
        train_size=relative_val_size,
        stratify=temp_groups[label_column],
        random_state=random_state,
    )

    split_map = {}

    for group in train_groups[group_column]:
        split_map[group] = "train"

    for group in val_groups[group_column]:
        split_map[group] = "val"

    for group in test_groups[group_column]:
        split_map[group] = "test"

    split_df = df.copy()
    split_df["split"] = split_df[group_column].map(split_map)

    if split_df["split"].isna().any():
        raise ValueError("Some rows were not assigned to a split.")

    return split_df


def summarize_split_balance(
    df: pd.DataFrame,
    split_column: str,
    label_column: str,
) -> pd.DataFrame:
    """
    Summarize class balance per split.
    """
    summary = (
        df.groupby([split_column, label_column])
        .size()
        .rename("count")
        .reset_index()
    )

    total_per_split = summary.groupby(split_column)["count"].transform("sum")
    summary["percentage"] = (summary["count"] / total_per_split * 100).round(2)

    return summary