"""Visualization helpers for experiment results."""

from __future__ import annotations

from typing import Dict

import matplotlib.pyplot as plt
import pandas as pd


def results_to_dataframe(results: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    """Convert {dataset: metrics_dict} to a DataFrame sorted by Macro-F1.

    Automatically adds imbalance_ratio from all_class_distributions.json when available.
    """
    from src.data_utils import load_class_distribution

    enriched = {}
    for key, metrics in results.items():
        row = dict(metrics)
        parts = key.split("/", 1)
        if len(parts) == 2:
            try:
                dist = load_class_distribution(parts[0], parts[1])
                row["imbalance_ratio"] = dist["imbalance_ratio"]
            except FileNotFoundError:
                pass
        enriched[key] = row

    df = pd.DataFrame(enriched).T
    df.index.name = "dataset"
    return df.sort_values("macro_f1", ascending=True)


def plot_metrics(results: Dict[str, Dict[str, float]], title: str = "", figsize=(18, 5)):
    """Horizontal bar charts of Macro-F1, TPRGap, and minority ratio per dataset."""
    df = results_to_dataframe(results)
    has_imbalance = "imbalance_ratio" in df.columns

    ncols = 3 if has_imbalance else 2
    fig, axes = plt.subplots(1, ncols, figsize=figsize)
    if title:
        fig.suptitle(title, fontsize=13, fontweight="bold")

    df["macro_f1"].plot.barh(ax=axes[0], color="steelblue", edgecolor="white")
    axes[0].set_title("Macro-F1  (higher is better)")
    axes[0].set_xlim(0, 1)
    axes[0].axvline(0.5, color="gray", linestyle="--", linewidth=0.8)
    for bar, val in zip(axes[0].patches, df["macro_f1"]):
        axes[0].text(val + 0.01, bar.get_y() + bar.get_height() / 2,
                     f"{val:.3f}", va="center", fontsize=8)

    df["tpr_gap"].plot.barh(ax=axes[1], color="tomato", edgecolor="white")
    axes[1].set_title("TPR Gap  (lower is better)")
    axes[1].set_xlim(0, 1)
    for bar, val in zip(axes[1].patches, df["tpr_gap"]):
        axes[1].text(val + 0.01, bar.get_y() + bar.get_height() / 2,
                     f"{val:.3f}", va="center", fontsize=8)

    if has_imbalance:
        df["imbalance_ratio"].plot.barh(ax=axes[2], color="mediumseagreen", edgecolor="white")
        axes[2].set_title("Imbalance Ratio  (majority / minority, 1 = balanced)")
        axes[2].set_xlim(0, df["imbalance_ratio"].max() * 1.15)
        axes[2].axvline(1, color="gray", linestyle="--", linewidth=0.8)
        for bar, val in zip(axes[2].patches, df["imbalance_ratio"]):
            axes[2].text(val + 0.05, bar.get_y() + bar.get_height() / 2,
                         f"{val:.1f}x", va="center", fontsize=8)

    plt.tight_layout()
