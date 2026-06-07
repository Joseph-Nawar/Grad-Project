import pandas as pd


def extract_false_positives(
    predictions_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Healthy patients predicted as Stroke.
    """
    return predictions_df[
        (predictions_df["true_label"] == "NonStroke")
        &
        (predictions_df["predicted_label"] == "Stroke")
    ].copy()

def extract_false_negatives(
    predictions_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Stroke patients predicted as healthy.

    Most dangerous error type.
    """
    return predictions_df[
        (predictions_df["true_label"] == "Stroke")
        &
        (predictions_df["predicted_label"] == "NonStroke")
    ].copy()

def extract_low_confidence_correct(
    predictions_df: pd.DataFrame,
    threshold: float = 0.10,
) -> pd.DataFrame:
    """
    Correct predictions made very close to 0.5.

    Indicates uncertainty.
    """
    df = predictions_df.copy()

    df = df[
        df["is_correct"]
    ]

    df["distance_from_threshold"] = (
        df["stroke_probability"] - 0.5
    ).abs()

    return (
        df
        .sort_values("distance_from_threshold")
        .head(50)
    )

def extract_high_confidence_errors(
    predictions_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Wrong predictions made with strong confidence.

    These often reveal:
    - label problems
    - dataset issues
    - systematic model weaknesses
    """
    df = predictions_df.copy()

    df = df[
        ~df["is_correct"]
    ]

    df["confidence"] = (
        df["stroke_probability"] - 0.5
    ).abs()

    return (
        df
        .sort_values(
            "confidence",
            ascending=False,
        )
    )

