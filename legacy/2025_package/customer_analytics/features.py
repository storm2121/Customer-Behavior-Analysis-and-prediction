"""Feature engineering.

These are deliberately pure functions over a DataFrame: no file access, no
globals, no plotting. That is what makes them the one part of the pipeline
that can be unit tested without the bank datasets.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


def days_since_last_transaction(merged: pd.DataFrame, today: pd.Timestamp | None = None) -> pd.Series:
    """Whole days between each client's last transaction and `today`.

    `today` is injectable so tests are not time-dependent; production callers
    leave it out and get the current date.
    """
    today = today if today is not None else pd.to_datetime("today")
    dates = pd.to_datetime(merged["date_transaction"], errors="coerce")
    days = (today - dates).dt.days
    return days.fillna(days.mean())


def avg_transaction_amount(merged: pd.DataFrame) -> pd.Series:
    """Total transacted per year of tenure.

    Zero-tenure clients yield an infinity from the division rather than NaN,
    so those are converted before the mean fill — otherwise a single new
    client poisons the column mean.
    """
    avg = merged["montant_local"] / merged["anciennete_client"]
    # np.nan (not pd.NA) keeps the column float; pd.NA would upcast to object
    # and make the fillna below a deprecated downcast.
    avg = avg.replace([np.inf, -np.inf], np.nan)
    return avg.fillna(avg.mean())


def age_group(merged: pd.DataFrame) -> pd.Series:
    """Bucket ages into the reporting bands used across the visualisations."""
    return pd.cut(merged["age"], bins=config.AGE_BINS, labels=config.AGE_LABELS)


def engineer(merged: pd.DataFrame) -> pd.DataFrame:
    """Attach every derived column the model and plots depend on."""
    merged = merged.copy()
    merged["days_since_last_transaction"] = days_since_last_transaction(merged)
    merged["avg_transaction_amount"] = avg_transaction_amount(merged)
    merged["age_group"] = age_group(merged)
    return merged
