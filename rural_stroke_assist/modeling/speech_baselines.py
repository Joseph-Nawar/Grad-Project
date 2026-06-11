import joblib
import pandas as pd
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer


NON_FEATURE_COLUMNS = {
    "path",
    "file_name",
    "label",
    "label_encoded",
    "speaker_id",
    "torgo_group",
    "split",
}


def get_speech_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return numeric feature columns for speech modeling."""
    return [
        column
        for column in df.columns
        if column not in NON_FEATURE_COLUMNS
    ]


def build_random_forest_speech_pipeline(
    random_state: int = 42,
) -> Pipeline:
    """
    Build MVP speech baseline pipeline.

    Includes:
    - median imputation
    - standard scaling
    - random forest classifier
    """
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=300,
                    class_weight="balanced",
                    random_state=random_state,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def save_model(model: Pipeline, output_path: Path) -> None:
    """Save trained sklearn model."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_path)


def load_model(model_path: Path) -> Pipeline:
    """Load trained sklearn model."""
    return joblib.load(model_path)