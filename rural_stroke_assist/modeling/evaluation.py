from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate_binary_classifier(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray | None = None,
) -> dict:
    """
    Evaluate binary classifier predictions.

    y_score should be the model probability for the positive class.
    """
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }

    if y_score is not None:
        metrics["roc_auc"] = roc_auc_score(y_true, y_score)

    return metrics


def create_classification_report_dict(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_names: list[str],
) -> dict:
    """
    Return sklearn classification report as a dictionary.
    """
    return classification_report(
        y_true,
        y_pred,
        target_names=target_names,
        output_dict=True,
        zero_division=0,
    )


def save_confusion_matrix_plot(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
    output_path: Path,
) -> None:
    """
    Save a confusion matrix plot.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    matrix = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(matrix)

    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))

    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)

    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Confusion Matrix")

    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, matrix[i, j], ha="center", va="center")

    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def save_markdown_results(
    metrics: dict,
    report: dict,
    output_path: Path,
    title: str,
) -> None:
    """
    Save evaluation results as a markdown file.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        f"# {title}",
        "",
        "## Summary Metrics",
        "",
    ]

    for key, value in metrics.items():
        lines.append(f"- {key}: {value:.4f}")

    lines.extend(
        [
            "",
            "## Classification Report",
            "",
            "| Class | Precision | Recall | F1-score | Support |",
            "|---|---:|---:|---:|---:|",
        ]
    )

    for label, values in report.items():
        if isinstance(values, dict) and "precision" in values:
            lines.append(
                f"| {label} | {values['precision']:.4f} | "
                f"{values['recall']:.4f} | {values['f1-score']:.4f} | "
                f"{int(values['support'])} |"
            )

    output_path.write_text("\n".join(lines), encoding="utf-8")