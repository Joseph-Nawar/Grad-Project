import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


def create_risk_deciles(
    y_true,
    y_score,
    n_bins: int = 10,
) -> pd.DataFrame:
    """
    Create risk deciles based on predicted probability.

    Decile 10 = highest predicted risk.
    """
    df = pd.DataFrame(
        {
            "y_true": y_true,
            "risk_score": y_score,
        }
    )

    df["risk_decile"] = pd.qcut(
        df["risk_score"],
        q=n_bins,
        labels=False,
        duplicates="drop",
    ) + 1

    return df


def summarize_risk_deciles(
    risk_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Summarize observed stroke rate per risk decile.
    """
    summary = (
        risk_df.groupby("risk_decile")
        .agg(
            patient_count=("y_true", "count"),
            stroke_count=("y_true", "sum"),
            observed_stroke_rate=("y_true", "mean"),
            mean_predicted_risk=("risk_score", "mean"),
            min_predicted_risk=("risk_score", "min"),
            max_predicted_risk=("risk_score", "max"),
        )
        .reset_index()
    )

    return summary.sort_values("risk_decile", ascending=True)


def compute_lift_table(
    risk_summary_df: pd.DataFrame,
    baseline_rate: float,
) -> pd.DataFrame:
    """
    Compute lift relative to the dataset baseline stroke rate.
    """
    lift_df = risk_summary_df.copy()

    lift_df["baseline_stroke_rate"] = baseline_rate
    lift_df["lift"] = lift_df["observed_stroke_rate"] / baseline_rate

    return lift_df


def evaluate_risk_ranking(
    y_true,
    y_score,
) -> dict:
    """
    Evaluate risk score quality for ranking and probability quality.
    """
    return {
        "roc_auc": roc_auc_score(y_true, y_score),
        "average_precision": average_precision_score(y_true, y_score),
        "brier_score": brier_score_loss(y_true, y_score),
        "baseline_positive_rate": float(pd.Series(y_true).mean()),
    }