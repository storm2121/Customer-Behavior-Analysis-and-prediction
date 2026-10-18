"""Typed contracts for the five extracts.

Two layers, on purpose. The *contract* (this module) states what the pipeline relies on: every
column present in order, every value parseable as its type, identifiers present on event rows,
dates in the window format, amounts non-negative. A contract violation stops the ingest. The
*data-quality checks* (``strata.ingest.dq``) measure everything that may legitimately vary
(unknown labels, nulls in descriptive columns, implausible ages, exact repeats) and report counts.

Validation errors are summarised as ``(column, check, count)`` triples. pandera's own error objects
carry the failing values, which on a private run are real data, so they never leave this module.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Any

import pandas as pd
import pandera.pandas as pa
from pandera.errors import SchemaErrors

from strata.contracts.specs import AGGREGATE, BRANCH, CARD, CLIENTS, ONLINE, Extract, ExtractSpec

CONTRACT_VERSION = "1"


class ColumnType(enum.StrEnum):
    """How a column is parsed from the text file."""

    ID = "id"
    INT = "int"
    FLOAT = "float"
    DATE = "date"
    TEXT = "text"


PANDAS_DTYPES: dict[ColumnType, str] = {
    ColumnType.ID: "str",
    ColumnType.INT: "Int64",
    ColumnType.FLOAT: "float64",
    ColumnType.DATE: "datetime64[us]",
    ColumnType.TEXT: "str",
}


def cast_to_contract(series: pd.Series, kind: ColumnType) -> pd.Series:
    """Cast a column to the contract dtype of ``kind``.

    pandas-stubs types ``astype`` with string literals only, so the lookup goes through ``Any``.
    """
    dtype: Any = PANDAS_DTYPES[kind]
    return series.astype(dtype)


_AGGREGATE_FLOATS = frozenset(
    {
        "moyenne_operation",
        "ecart_type_operation",
        "mediane_operation",
        "minimum_operation",
        "maximum_operation",
        "moyenne_montant_transaction",
        "ecart_type_montant_transaction",
        "mediane_montant_transaction",
        "minimum_montant_transaction",
        "maximum_montant_transaction",
    }
)

COLUMN_TYPES: dict[Extract, dict[str, ColumnType]] = {
    Extract.CLIENTS: {
        "identifiant_client": ColumnType.ID,
        **dict.fromkeys(CLIENTS.columns[1:], ColumnType.INT),
    },
    Extract.CARD: {
        "identifiant_client": ColumnType.ID,
        "date_transaction": ColumnType.DATE,
        "montant_local": ColumnType.FLOAT,
        "libelle_nature_transaction": ColumnType.TEXT,
        "type_origine": ColumnType.TEXT,
    },
    Extract.ONLINE: {
        "identifiant_client": ColumnType.ID,
        "service": ColumnType.TEXT,
        "operation_date_time": ColumnType.DATE,
        "montant_transaction": ColumnType.FLOAT,
    },
    Extract.BRANCH: {
        "identifiant_client": ColumnType.ID,
        "date_operation": ColumnType.DATE,
        "montant_operation": ColumnType.FLOAT,
        "sens_operation": ColumnType.TEXT,
        "operation": ColumnType.TEXT,
        "devise": ColumnType.TEXT,
    },
    Extract.AGGREGATE: {
        column: (
            ColumnType.ID
            if column == "identifier"
            else ColumnType.TEXT
            if column == "a_email"
            else ColumnType.FLOAT
            if column in _AGGREGATE_FLOATS
            else ColumnType.INT
        )
        for column in AGGREGATE.columns
    },
}

AMOUNT_COLUMNS: dict[Extract, str] = {
    Extract.CARD: "montant_local",
    Extract.ONLINE: "montant_transaction",
    Extract.BRANCH: "montant_operation",
}

_NON_NEGATIVE_INTS = frozenset(
    {"anciennete_client", "age", "nombre_d_enfants", "duree_creation_client_en_annee"}
)


def _column(spec: ExtractSpec, name: str, kind: ColumnType) -> pa.Column:
    checks: list[pa.Check] = []
    nullable = True
    unique = False
    if kind is ColumnType.ID:
        # Event rows and client rows must carry an identifier; the aggregate file interleaves
        # empty rows, so its identifier column alone may be null, and is unique where present
        # (pandera's built-in uniqueness would count the empty rows as duplicates).
        nullable = spec.name is Extract.AGGREGATE
        unique = spec.name is Extract.CLIENTS
        if spec.name is Extract.AGGREGATE:
            checks.append(
                pa.Check(
                    lambda s: ~(s.duplicated(keep=False) & s.notna()), name="unique_when_present"
                )
            )
    elif kind is ColumnType.DATE:
        nullable = False
    elif kind is ColumnType.FLOAT and name in AMOUNT_COLUMNS.values():
        checks.append(pa.Check.ge(0, name="amount_non_negative"))
    elif kind is ColumnType.INT and name in _NON_NEGATIVE_INTS:
        checks.append(pa.Check.ge(0, name="non_negative"))
    if name == "age":
        checks.append(pa.Check.le(130, name="age_below_130"))
    return pa.Column(PANDAS_DTYPES[kind], checks=checks, nullable=nullable, unique=unique)


def schema_for(spec: ExtractSpec) -> pa.DataFrameSchema:
    """The pandera schema of one typed extract (columns strict and ordered)."""
    types = COLUMN_TYPES[spec.name]
    return pa.DataFrameSchema(
        {name: _column(spec, name, types[name]) for name in spec.columns},
        strict=True,
        ordered=True,
        name=spec.name.value,
        report_duplicates="all",
    )


SCHEMAS: dict[Extract, pa.DataFrameSchema] = {
    spec.name: schema_for(spec) for spec in (CLIENTS, CARD, ONLINE, BRANCH, AGGREGATE)
}


@dataclass(frozen=True, slots=True)
class Violation:
    """One failed rule, counted. Deliberately has no field for the failing values."""

    extract: str
    column: str
    check: str
    count: int

    def render(self) -> str:
        return f"{self.extract}.{self.column}: {self.check} failed for {self.count} row(s)"


class ContractError(ValueError):
    """An extract does not satisfy its contract. The message holds counts, never values."""

    def __init__(self, violations: list[Violation]) -> None:
        self.violations = violations
        lines = "; ".join(violation.render() for violation in violations)
        super().__init__(f"contract violated: {lines}")


def _check_name(item: object) -> str:
    check = getattr(item, "check", None)
    name = getattr(check, "name", None)
    if isinstance(name, str) and name:
        return name
    if isinstance(check, str) and check:
        return check
    reason = getattr(item, "reason_code", None)
    return str(getattr(reason, "name", reason or "unknown")).lower()


def summarise(extract: str, error: SchemaErrors) -> list[Violation]:
    """Count pandera's individual errors by column and check, dropping every value.

    Built from the structured ``schema_errors`` list, never from the error message: for some error
    classes (duplicates, for example) pandera writes the offending values into the message and
    leaves ``failure_cases`` empty.
    """
    counts: dict[tuple[str, str], int] = {}
    for item in error.schema_errors:
        column = str(getattr(item, "column_name", None) or "<frame>")
        cases = getattr(item, "failure_cases", None)
        size = len(cases) if isinstance(cases, pd.DataFrame | pd.Series) and len(cases) else 1
        key = (column, _check_name(item))
        counts[key] = counts.get(key, 0) + size
    return [
        Violation(extract=extract, column=column, check=check, count=count)
        for (column, check), count in sorted(counts.items())
    ]


def validate(frame: pd.DataFrame, spec: ExtractSpec) -> pd.DataFrame:
    """Validate a typed frame; raise :class:`ContractError` with counts on failure."""
    try:
        return SCHEMAS[spec.name].validate(frame, lazy=True)
    except SchemaErrors as error:
        raise ContractError(summarise(spec.name.value, error)) from None
