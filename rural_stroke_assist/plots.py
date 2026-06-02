from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def save_bar_plot(
    series: pd.Series,
    title: str,
    xlabel: str,
    ylabel: str,
    output_path: Path,
) -> None:
    """Save a simple bar plot."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))
    series.plot(kind="bar")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def save_histogram(
    values,
    title: str,
    xlabel: str,
    ylabel: str,
    output_path: Path,
    bins: int = 30,
) -> None:
    """Save a histogram."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.hist(values, bins=bins)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()