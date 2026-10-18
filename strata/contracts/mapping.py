"""Source mappings: how any extract of clients and dated channel events becomes canonical tables.

A source is described by a YAML file (``mappings/<source>.yaml``, ARCHITECTURE section 5.0): which
file plays which role, how it is encoded, delimited and dated, which source column feeds which
canonical column, and how the source's codes translate into the reference vocabulary of the seeds.
``strata ingest --mapping <file>`` loads the source into three canonical tables, and dbt staging
reads only those, so every mart, claim and model works unchanged on a new source:

- ``raw.clients``: ``client_id`` and the client attributes of :data:`CLIENT_TYPES`;
- ``raw.events``: one row per dated event, with the columns of :data:`EVENT_TYPES` plus
  ``source_file`` and ``source_row`` (the row's position in its file);
- ``raw.client_aggregates`` (optional): ``client_id`` and the attributes of :data:`AGGREGATE_TYPES`.

A file mapped with ``contract`` is read through the typed contract of the reference extract
(``strata.contracts.specs``), exactly as before mappings existed, so the reference instance keeps
its manifest, its data-quality report and its marts. Any other file is read generically: the mapped
columns must be present and parse to their canonical types, or the ingest stops.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from strata.config import find_repo_root
from strata.contracts.extracts import ColumnType, ContractError, Violation, cast_to_contract
from strata.contracts.specs import DATE_FORMAT, ENCODING, Extract

Channel = Literal["card", "online", "branch"]
CHANNELS: tuple[Channel, ...] = ("card", "online", "branch")

MAPPINGS_DIR = Path("mappings")
REFERENCE_MAPPING = MAPPINGS_DIR / "reference_cih.yaml"
SYNTHETIC_MAPPING = MAPPINGS_DIR / "synthetic.yaml"

EVENT_TYPES: dict[str, ColumnType] = {
    "client_id": ColumnType.ID,
    "ts": ColumnType.DATE,
    "channel": ColumnType.TEXT,
    "kind": ColumnType.TEXT,
    "amount": ColumnType.FLOAT,
    "direction": ColumnType.TEXT,
    "origin": ColumnType.TEXT,
    "currency": ColumnType.TEXT,
}
"""Canonical event columns. ``kind`` holds the reference vocabulary's label for the event's
category (a card category, an online service, a branch operation); ``direction`` and ``origin``
are optional; ``currency`` is an extension of ARCHITECTURE section 5.0 (ADR 0010)."""

EVENT_COLUMNS = (*EVENT_TYPES, "source_file", "source_row")

CLIENT_TYPES: dict[str, ColumnType] = {
    "client_id": ColumnType.ID,
    "tenure_years": ColumnType.INT,
    "age": ColumnType.INT,
    "gender_code": ColumnType.INT,
    "marital_code": ColumnType.INT,
    "children": ColumnType.INT,
    "sector_code": ColumnType.INT,
    "profession_code": ColumnType.INT,
    "socio_category_code": ColumnType.INT,
}

_AGGREGATE_INTS = [
    "tenure_years",
    "age",
    "civilite_code",
    "birth_country_code",
    "residence_country_code",
    "marital_code",
    "children",
    "professional_status_code",
    "creation_place_code",
]
_AGGREGATE_REST: tuple[tuple[str, ColumnType], ...] = (
    ("email_flag", ColumnType.TEXT),
    ("resident_status_code", ColumnType.INT),
    ("profession_flag", ColumnType.INT),
    ("online_flag", ColumnType.INT),
    ("active_contract_flag", ColumnType.INT),
    *(
        (name, ColumnType.INT)
        for name in [
            "ops_one_off_transfer",
            "ops_bill_pay",
            "ops_standing_transfer",
            "ops_card_service",
            "ops_cash_express",
            "ops_eshopping_recharge",
            "ops_card_debit",
            "ops_binatna_recharge",
            "ops_cheque_image",
            "ops_total",
        ]
    ),
    *(
        (name, ColumnType.FLOAT)
        for name in [
            "op_amount_mean",
            "op_amount_std",
            "op_amount_median",
            "op_amount_min",
            "op_amount_max",
        ]
    ),
    *(
        (name, ColumnType.INT)
        for name in [
            "months_since_first_op",
            "months_since_last_op",
            "credits_total",
            "credits_mortgage",
            "credits_consumer",
            "debit_credit_flag",
            "card_cih_atm_count",
            "card_merchant_national_count",
            "card_merchant_international_count",
            "other_national_card_atm_count",
            "other_international_card_atm_count",
            "txn_deposit",
            "txn_withdrawal",
            "txn_mastercard_send",
            "txn_refund",
            "txn_bill_payment",
            "txn_atm_purchase",
            "txn_visa_load",
            "txn_ecommerce",
            "txn_purchases",
            "txn_vignette",
            "txn_total",
        ]
    ),
    *(
        (name, ColumnType.FLOAT)
        for name in [
            "txn_amount_mean",
            "txn_amount_std",
            "txn_amount_median",
            "txn_amount_min",
            "txn_amount_max",
        ]
    ),
    *(
        (name, ColumnType.INT)
        for name in ["international_txn_flag", "months_since_first_txn", "months_since_last_txn"]
    ),
)
AGGREGATE_TYPES: dict[str, ColumnType] = {
    "client_id": ColumnType.ID,
    **dict.fromkeys(_AGGREGATE_INTS, ColumnType.INT),
    **dict(_AGGREGATE_REST),
}


class _Strict(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class TableFile(_Strict):
    """A client-grain file: canonical column name to source column name."""

    file: str
    columns: dict[str, str]
    contract: Extract | None = None
    encoding: str | None = None
    delimiter: str | None = None
    decimal: str | None = None


class EventFile(_Strict):
    """An event file of one channel, or of several told apart by ``channel_column``."""

    file: str
    columns: dict[str, str]
    channel: Channel | None = None
    channel_column: str | None = None
    channel_codes: dict[str, Channel] = Field(default_factory=dict)
    kind_codes: dict[str, str] = Field(default_factory=dict)
    direction_codes: dict[str, str] = Field(default_factory=dict)
    contract: Extract | None = None
    encoding: str | None = None
    delimiter: str | None = None
    decimal: str | None = None
    date_format: str | None = None

    @model_validator(mode="after")
    def _channel_is_known(self) -> EventFile:
        if (self.channel is None) == (self.channel_column is None):
            msg = f"{self.file}: give either `channel` or `channel_column`, not both"
            raise ValueError(msg)
        if self.channel_column is not None and not self.channel_codes:
            msg = f"{self.file}: `channel_column` needs `channel_codes`"
            raise ValueError(msg)
        unknown = sorted(set(self.columns) - set(EVENT_TYPES) - {"channel"})
        if unknown:
            msg = f"{self.file}: not canonical event columns: {unknown}"
            raise ValueError(msg)
        for required in ("client_id", "ts"):
            if required not in self.columns:
                msg = f"{self.file}: an event file needs a `{required}` column"
                raise ValueError(msg)
        return self


class SourceMapping(_Strict):
    """A whole source: its defaults and every file's role."""

    source: str
    description: str
    encoding: str = "utf-8"
    delimiter: str = ","
    decimal: str = "."
    date_format: str = "%Y-%m-%d"
    clients: TableFile
    events: list[EventFile] = Field(min_length=1)
    client_aggregates: TableFile | None = None

    @model_validator(mode="after")
    def _canonical_names(self) -> SourceMapping:
        for role, table, types in (
            ("clients", self.clients, CLIENT_TYPES),
            ("client_aggregates", self.client_aggregates, AGGREGATE_TYPES),
        ):
            if table is None:
                continue
            unknown = sorted(set(table.columns) - set(types))
            if unknown:
                msg = f"{role}: not canonical columns: {unknown}"
                raise ValueError(msg)
            if "client_id" not in table.columns:
                msg = f"{role}: a client file needs a `client_id` column"
                raise ValueError(msg)
        return self

    @property
    def files(self) -> list[str]:
        """Every file the source consists of."""
        tables = [self.clients, *([self.client_aggregates] if self.client_aggregates else [])]
        return [table.file for table in tables] + [event.file for event in self.events]


class LoadedMapping(BaseModel):
    """A mapping with where it came from: the digest travels into the manifest and the ledger."""

    model_config = ConfigDict(frozen=True)

    mapping: SourceMapping
    path: str
    sha256: str


CODE_ROOT = Path(__file__).resolve().parents[2]
"""The checkout the package runs from, which ships the reference and synthetic mappings."""


def mapping_path(name: Path | str, root: Path | None = None) -> Path:
    """``name`` as a path; a bare name such as ``synthetic`` means ``mappings/synthetic.yaml``.

    A relative path is looked up under ``root`` (the configured repository) and then under the
    checkout the code runs from, which ships the mappings of the reference instance.
    """
    path = Path(name)
    if path.suffix not in {".yaml", ".yml"}:
        path = MAPPINGS_DIR / f"{path}.yaml"
    if path.is_absolute():
        return path
    candidate = (root or find_repo_root()) / path
    return (
        candidate if candidate.is_file() or not (CODE_ROOT / path).is_file() else CODE_ROOT / path
    )


def load_mapping(name: Path | str, root: Path | None = None) -> LoadedMapping:
    """Read and validate a mapping file."""
    path = mapping_path(name, root)
    payload = path.read_bytes()
    mapping = SourceMapping.model_validate(yaml.safe_load(payload.decode("utf-8")))
    return LoadedMapping(
        mapping=mapping, path=path.name, sha256=hashlib.sha256(payload).hexdigest()
    )


def file_settings(mapping: SourceMapping, file: TableFile | EventFile) -> dict[str, str]:
    """Encoding, delimiter, decimal mark and date format of ``file`` (its own, or the source's)."""
    date_format = getattr(file, "date_format", None)
    return {
        "encoding": file.encoding or mapping.encoding,
        "delimiter": file.delimiter or mapping.delimiter,
        "decimal": file.decimal or mapping.decimal,
        "date_format": date_format or mapping.date_format,
    }


def source_columns(
    file: TableFile | EventFile, types: dict[str, ColumnType]
) -> dict[str, ColumnType]:
    """The source columns to read and the canonical type each one is parsed to."""
    wanted = {source: types[canonical] for canonical, source in file.columns.items()}
    if isinstance(file, EventFile) and file.channel_column is not None:
        wanted[file.channel_column] = ColumnType.TEXT
    return wanted


def _empty(kind: ColumnType, length: int) -> pd.Series:
    return cast_to_contract(pd.Series([None] * length, dtype="object"), kind)


def canonical_table(
    frame: pd.DataFrame, file: TableFile, types: dict[str, ColumnType]
) -> pd.DataFrame:
    """A client-grain frame in canonical columns; unmapped attributes are empty."""
    columns = {
        name: (
            cast_to_contract(frame[file.columns[name]].reset_index(drop=True), kind)
            if name in file.columns
            else _empty(kind, len(frame))
        )
        for name, kind in types.items()
    }
    return pd.DataFrame(columns)


def _recode(values: pd.Series, codes: dict[str, str], label: str, column: str) -> pd.Series:
    """Translate source codes; a value without a translation stops the ingest."""
    if not codes:
        return values
    present = values.notna()
    unknown = present & ~values.astype("object").isin(list(codes))
    if unknown.any():
        raise ContractError([Violation(label, column, "codes_mapped", int(unknown.sum()))])
    return values.astype("object").map(codes).where(present)


def canonical_events(frame: pd.DataFrame, file: EventFile, *, label: str) -> pd.DataFrame:
    """An event frame in canonical columns, with ``source_file`` and ``source_row``."""
    length = len(frame)
    frame = frame.reset_index(drop=True)
    if file.channel is not None:
        channel = pd.Series([file.channel] * length, dtype="object")
    else:
        column = file.channel_column or ""
        channel = _recode(frame[column], dict(file.channel_codes), label, column)
        if channel.isna().any():
            raise ContractError(
                [Violation(label, column, "channel_present", int(channel.isna().sum()))]
            )
    columns: dict[str, pd.Series] = {}
    for name, kind in EVENT_TYPES.items():
        if name == "channel":
            columns[name] = cast_to_contract(channel, kind)
        elif name in file.columns:
            values = frame[file.columns[name]]
            if name == "kind":
                values = _recode(values, dict(file.kind_codes), label, file.columns[name])
            if name == "direction":
                values = _recode(values, dict(file.direction_codes), label, file.columns[name])
            columns[name] = cast_to_contract(values, kind)
        else:
            columns[name] = _empty(kind, length)
    canonical = pd.DataFrame(columns)
    canonical["source_file"] = pd.Series([file.file] * length, dtype="string")
    canonical["source_row"] = np.arange(1, length + 1, dtype=np.int64)
    return canonical


__all__ = [
    "AGGREGATE_TYPES",
    "CHANNELS",
    "CLIENT_TYPES",
    "DATE_FORMAT",
    "ENCODING",
    "EVENT_COLUMNS",
    "EVENT_TYPES",
    "REFERENCE_MAPPING",
    "SYNTHETIC_MAPPING",
    "EventFile",
    "LoadedMapping",
    "SourceMapping",
    "TableFile",
    "canonical_events",
    "canonical_table",
    "file_settings",
    "load_mapping",
    "mapping_path",
    "source_columns",
]
