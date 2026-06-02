import pandas as pd


def summarize_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Return missing-value counts and percentages."""
    missing_count = df.isna().sum()
    missing_percentage = (df.isna().mean() * 100).round(2)

    return pd.DataFrame(
        {
            "missing_count": missing_count,
            "missing_percentage": missing_percentage,
        }
    ).sort_values("missing_percentage", ascending=False)


def get_categorical_columns(df: pd.DataFrame) -> list[str]:
    """Return categorical columns."""
    return df.select_dtypes(include=["object", "category"]).columns.tolist()


def get_numeric_columns(df: pd.DataFrame) -> list[str]:
    """Return numeric columns."""
    return df.select_dtypes(include=["number"]).columns.tolist()


def summarize_class_balance(df: pd.DataFrame, target_column: str) -> pd.DataFrame:
    """Return class balance for a target column."""
    counts = df[target_column].value_counts(dropna=False)
    percentages = (df[target_column].value_counts(normalize=True, dropna=False) * 100).round(2)

    return pd.DataFrame(
        {
            "count": counts,
            "percentage": percentages,
        }
    )