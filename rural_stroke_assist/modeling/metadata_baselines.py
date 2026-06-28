from pathlib import Path

import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from rural_stroke_assist.preprocessing.metadata import (
    build_metadata_preprocessor,
)


def build_logistic_regression_metadata_pipeline(
    random_state: int = 42,
) -> Pipeline:
    """
    Build interpretable metadata baseline.

    Logistic Regression is a strong first healthcare baseline because it is
    simple, transparent, and works well for tabular risk-factor data.
    """
    return Pipeline(
        steps=[
            ("preprocessor", build_metadata_preprocessor(scale_numeric=True)),
            (
                "model",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=1000,
                    random_state=random_state,
                ),
            ),
        ]
    )


def build_random_forest_metadata_pipeline(
    random_state: int = 42,
) -> Pipeline:
    """
    Build tree-based metadata baseline.

    Random Forest handles nonlinear feature interactions and does not require
    numeric scaling, but we keep preprocessing consistent.
    """
    return Pipeline(
        steps=[
            ("preprocessor", build_metadata_preprocessor(scale_numeric=False)),
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


def save_metadata_model(model: Pipeline, output_path: Path) -> None:
    """Save fitted metadata model pipeline."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_path)


def load_metadata_model(model_path: Path) -> Pipeline:
    """Load fitted metadata model pipeline."""
    return joblib.load(model_path)


import pandas as pd
from sklearn.inspection import permutation_importance


def compute_metadata_permutation_importance(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    scoring: str = "roc_auc",
    n_repeats: int = 20,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Compute permutation feature importance for the fitted metadata model.

    This is model-agnostic and works with sklearn Pipelines.
    """
    result = permutation_importance(
        model,
        X_test,
        y_test,
        scoring=scoring,
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=-1,
    )

    importance_df = pd.DataFrame(
        {
            "feature": X_test.columns,
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        }
    )

    return importance_df.sort_values(
        by="importance_mean",
        ascending=False,
    ).reset_index(drop=True)