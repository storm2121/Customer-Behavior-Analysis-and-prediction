"""Plots.

Every function takes an explicit `show` flag so the CLI can render figures
interactively while tests and batch runs import the module without a display.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from . import config


def _finish(show: bool) -> None:
    if show:
        plt.show()
    else:
        plt.close()


def payments_by_age_group(merged: pd.DataFrame, show: bool = True) -> None:
    plt.figure(figsize=(10, 6))
    sns.countplot(data=merged, x="age_group", order=config.AGE_LABELS)
    plt.title("Frequency of Digital Payments by Age Group")
    plt.xlabel("Age Group")
    plt.ylabel("Number of Clients")
    _finish(show)


def payments_by_gender(merged: pd.DataFrame, show: bool = True) -> None:
    plt.figure(figsize=(6, 6))
    sns.countplot(data=merged, x="genre")
    plt.title("Frequency of Digital Payments by Gender")
    plt.xlabel("Gender")
    plt.ylabel("Number of Clients")
    _finish(show)


def average_amount_by_age_group(merged: pd.DataFrame, show: bool = True) -> None:
    averages = merged.groupby("age_group", observed=False)["avg_transaction_amount"].mean().reset_index()
    plt.figure(figsize=(10, 6))
    sns.barplot(data=averages, x="age_group", y="avg_transaction_amount", order=config.AGE_LABELS)
    plt.title("Average Transaction Amount by Age Group")
    plt.xlabel("Age Group")
    plt.ylabel("Average Transaction Amount")
    _finish(show)


def transaction_recency(merged: pd.DataFrame, show: bool = True) -> None:
    plt.figure(figsize=(10, 6))
    sns.histplot(data=merged, x="days_since_last_transaction", bins=30, kde=True)
    plt.title("Transaction Frequency Pattern")
    plt.xlabel("Days Since Last Transaction")
    plt.ylabel("Number of Clients")
    _finish(show)


def elbow(scores: list[float], show: bool = True) -> None:
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(scores) + 1), scores, marker="o")
    plt.title("Elbow Method for Optimal k")
    plt.xlabel("Number of Clusters")
    plt.ylabel("WCSS")
    _finish(show)


def clusters(components: pd.DataFrame, show: bool = True) -> None:
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=components, x="PC1", y="PC2", hue="cluster", palette="Set1")
    plt.title("Customer Segments (PCA projection)")
    _finish(show)


def overview(merged: pd.DataFrame, show: bool = True) -> None:
    """Render the full exploratory set in one call."""
    payments_by_age_group(merged, show)
    payments_by_gender(merged, show)
    average_amount_by_age_group(merged, show)
    transaction_recency(merged, show)
