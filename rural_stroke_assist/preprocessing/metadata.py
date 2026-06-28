import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


TARGET_COLUMN = "stroke"

ID_COLUMNS = ["id"]

NUMERIC_FEATURES = [
    "age",
    "hypertension",
    "heart_disease",
    "avg_glucose_level",
    "bmi",
]

CATEGORICAL_FEATURES = [
    "gender",
    "ever_married",
    "work_type",
    "Residence_type",
    "smoking_status",
]

FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def get_metadata_feature_columns() -> list[str]:
    """Return the approved metadata feature columns for MVP modeling."""
    return FEATURE_COLUMNS.copy()


def validate_metadata_columns(df: pd.DataFrame) -> None:
    """
    Validate that required metadata columns exist.
    """
    required_columns = set(FEATURE_COLUMNS + [TARGET_COLUMN])
    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(f"Missing metadata columns: {sorted(missing_columns)}")


def prepare_metadata_modeling_frame(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only approved metadata features, target, and split column.

    Drops:
    - id
    - any unexpected columns
    """
    validate_metadata_columns(df)

    columns_to_keep = FEATURE_COLUMNS + [TARGET_COLUMN]

    if "split" in df.columns:
        columns_to_keep.append("split")

    return df[columns_to_keep].copy()


def build_metadata_preprocessor(
    scale_numeric: bool = True,
) -> ColumnTransformer:
    """
    Build preprocessing pipeline for metadata features.

    Numeric:
    - median imputation
    - optional scaling

    Categorical:
    - most-frequent imputation
    - one-hot encoding
    """
    numeric_steps = [
        ("imputer", SimpleImputer(strategy="median")),
    ]

    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))

    numeric_pipeline = Pipeline(steps=numeric_steps)

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )


def split_metadata_features_target(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Split metadata dataframe into X and y.
    """
    validate_metadata_columns(df)

    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].copy()

    return X, y