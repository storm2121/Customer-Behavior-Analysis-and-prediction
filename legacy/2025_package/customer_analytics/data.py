"""Loading, sampling, aggregation and merging of the five source extracts."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from . import config

logger = logging.getLogger(__name__)


def load_raw(data_dir: Path | None = None) -> dict[str, pd.DataFrame]:
    """Read every configured CSV into a frame keyed by short name.

    Raises FileNotFoundError naming the missing file rather than failing deep
    inside pandas — the datasets are not distributed with this repository.
    """
    data_dir = Path(data_dir or config.DATA_DIR)
    frames: dict[str, pd.DataFrame] = {}

    for name, filename in config.CSV_FILES.items():
        path = data_dir / filename
        if not path.exists():
            raise FileNotFoundError(
                f"Missing dataset '{filename}' in {data_dir.resolve()}. "
                "See the Data section of README.md."
            )
        frames[name] = pd.read_csv(path, encoding=config.CSV_ENCODING)
        logger.info("Loaded %-18s %s", name, frames[name].shape)

    return frames


def sample_clients(frames: dict[str, pd.DataFrame], sample_size: int | None = None) -> dict[str, pd.DataFrame]:
    """Take a client sample and restrict every other frame to those clients.

    Filtering the satellite tables to the sampled ids keeps the later merges
    cheap and prevents aggregates being computed over clients we then discard.
    """
    sample_size = sample_size or config.SAMPLE_SIZE
    frames = {name: frame.copy() for name, frame in frames.items()}

    clients = frames["clients"]
    n = min(sample_size, len(clients))
    clients = clients.sample(n=n, random_state=config.RANDOM_STATE)
    frames["clients"] = clients

    # Ids arrive with mixed dtypes across extracts; string is the common ground.
    client_ids = clients[config.CLIENT_ID].astype(str).unique()

    for name in ("transactions", "online_operations", "agency_operations"):
        frame = frames[name]
        frame[config.CLIENT_ID] = frame[config.CLIENT_ID].astype(str)
        frames[name] = frame[frame[config.CLIENT_ID].isin(client_ids)]

    akhawayn = frames["al_akhawayn"]
    akhawayn["identifier"] = akhawayn["identifier"].astype(str)
    frames["al_akhawayn"] = akhawayn[akhawayn["identifier"].isin(client_ids)]

    frames["clients"][config.CLIENT_ID] = clients[config.CLIENT_ID].astype(str)
    logger.info("Sampled %d clients", n)
    return frames


def _mode_or_unknown(series: pd.Series) -> str:
    """Most common value, falling back to 'Unknown' for an all-null group."""
    mode = series.mode()
    return mode.iloc[0] if not mode.empty else "Unknown"


def aggregate_per_client(frames: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Collapse the per-event tables to one row per client."""
    transactions = (
        frames["transactions"]
        .groupby(config.CLIENT_ID)
        .agg(
            {
                "montant_local": "sum",
                config.TARGET: _mode_or_unknown,
                "date_transaction": "max",
            }
        )
        .reset_index()
    )

    online = (
        frames["online_operations"]
        .groupby(config.CLIENT_ID)
        .agg({"montant_transaction": "sum", "service": _mode_or_unknown})
        .reset_index()
    )

    agency = (
        frames["agency_operations"]
        .groupby(config.CLIENT_ID)
        .agg({"montant_operation": "sum", "operation": _mode_or_unknown})
        .reset_index()
    )

    return {"transactions": transactions, "online": online, "agency": agency}


def merge_all(frames: dict[str, pd.DataFrame], aggregates: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Left-join the aggregates and the Al Akhawayn demographics onto clients."""
    merged = frames["clients"].copy()
    merged = merged.merge(aggregates["transactions"], on=config.CLIENT_ID, how="left")
    merged = merged.merge(aggregates["online"], on=config.CLIENT_ID, how="left", suffixes=("", "_online"))
    merged = merged.merge(aggregates["agency"], on=config.CLIENT_ID, how="left", suffixes=("", "_agency"))

    merged = merged.merge(
        frames["al_akhawayn"][["identifier", "age", "situation_familiale"]],
        left_on=config.CLIENT_ID,
        right_on="identifier",
        how="left",
        suffixes=("", "_al"),
    )

    # The Al Akhawayn extract backfills demographics the bank export is missing.
    merged["age"] = merged["age"].fillna(merged["age_al"])
    merged["situation_familiale"] = merged["situation_familiale"].fillna(merged["situation_familiale_al"])
    merged = merged.drop(columns=["age_al", "situation_familiale_al", "identifier"])

    return merged


def validate(merged: pd.DataFrame) -> None:
    """Fail loudly if the merge dropped a column the rest of the pipeline needs."""
    missing = [col for col in config.REQUIRED_COLUMNS if col not in merged.columns]
    if missing:
        raise ValueError(f"Merge produced no {missing}; check the source extracts.")


def impute(merged: pd.DataFrame) -> pd.DataFrame:
    """Fill gaps: mode for categoricals, mean for numerics."""
    merged = merged.copy()

    for col in ("genre", "situation_familiale"):
        mode = merged[col].mode()
        if not mode.empty:
            merged[col] = merged[col].fillna(mode[0])

    for col in config.NUMERICAL_FEATURES:
        if col in merged.columns:
            merged[col] = merged[col].fillna(merged[col].mean())

    return merged


def build_dataset(data_dir: Path | None = None, sample_size: int | None = None) -> pd.DataFrame:
    """Run the whole load → sample → aggregate → merge → impute chain."""
    from .features import engineer

    frames = sample_clients(load_raw(data_dir), sample_size)
    merged = merge_all(frames, aggregate_per_client(frames))
    validate(merged)
    merged = engineer(merged)
    return impute(merged)
