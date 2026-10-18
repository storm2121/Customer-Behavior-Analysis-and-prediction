"""Read one extract from disk into a typed, contract-checked frame.

Every column is first read as text, so that parsing is explicit: an empty field is a missing value,
anything else must parse as the column's type or the contract fails. Counting lines independently of
the CSV parser catches the silent failure modes of ``read_csv`` (skipped blank lines, a quoted field
swallowing a line break).
"""

from __future__ import annotations

import hashlib
import io
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from kashf.contracts.extracts import (
    COLUMN_TYPES,
    ColumnType,
    ContractError,
    Violation,
    cast_to_contract,
    validate,
)
from kashf.contracts.specs import DATE_FORMAT, ENCODING, ExtractSpec


@dataclass(frozen=True, slots=True)
class ReadResult:
    """A typed frame plus the facts about the file it came from."""

    spec: ExtractSpec
    frame: pd.DataFrame
    sha256: str
    size_bytes: int
    data_lines: int
    line_terminator: str


def _line_terminator(text: str) -> str:
    crlf = text.count("\r\n")
    lf = text.count("\n")
    if lf == 0:
        return "none"
    if crlf == lf:
        return "CRLF"
    return "LF" if crlf == 0 else "mixed"


def _data_lines(text: str) -> int:
    lines = text.count("\n") + (0 if text.endswith("\n") else 1)
    return max(lines - 1, 0)


def parse_column(series: pd.Series, kind: ColumnType) -> tuple[pd.Series, int]:
    """Parse a text column; return the typed column and how many non-empty values failed."""
    text = series.astype("str")
    empty = text.eq("") | text.isna()
    if kind in {ColumnType.ID, ColumnType.TEXT}:
        return cast_to_contract(text.mask(empty), kind), 0
    if kind is ColumnType.DATE:
        parsed = pd.to_datetime(text.mask(empty), format=DATE_FORMAT, errors="coerce")
        failures = int((parsed.isna() & ~empty).sum())
        return cast_to_contract(parsed, kind), failures
    numbers = pd.to_numeric(text.mask(empty), errors="coerce")
    failures = int((numbers.isna() & ~empty).sum())
    if kind is ColumnType.INT:
        fractional = numbers.notna() & (numbers % 1 != 0)
        failures += int(fractional.sum())
        numbers = numbers.mask(fractional)
        return numbers.astype("Int64"), failures
    return numbers.astype(np.float64), failures


def parse_frame(raw: pd.DataFrame, spec: ExtractSpec) -> pd.DataFrame:
    """Turn an all-text frame into the contract's types; raise on any unparseable value."""
    types = COLUMN_TYPES[spec.name]
    columns: dict[str, pd.Series] = {}
    violations: list[Violation] = []
    for name in spec.columns:
        typed, failures = parse_column(raw[name], types[name])
        columns[name] = typed
        if failures:
            violations.append(
                Violation(spec.name.value, name, f"parse_{types[name].value}", failures)
            )
    if violations:
        raise ContractError(violations)
    return pd.DataFrame(columns)


def read_extract(path: Path, spec: ExtractSpec) -> ReadResult:
    """Read, parse and validate one extract file."""
    payload = path.read_bytes()
    text = payload.decode(ENCODING)
    header = text.split("\n", 1)[0].rstrip("\r").split(",")
    if tuple(header) != spec.columns:
        raise ContractError([Violation(spec.name.value, "<header>", "columns_match_spec", 1)])
    raw = pd.read_csv(
        io.StringIO(text), dtype=str, keep_default_na=False, na_filter=False, skip_blank_lines=False
    )
    lines = _data_lines(text)
    if len(raw) != lines:
        violation = Violation(spec.name.value, "<file>", "rows_equal_lines", abs(len(raw) - lines))
        raise ContractError([violation])
    frame = validate(parse_frame(raw, spec), spec)
    return ReadResult(
        spec=spec,
        frame=frame,
        sha256=hashlib.sha256(payload).hexdigest(),
        size_bytes=len(payload),
        data_lines=lines,
        line_terminator=_line_terminator(text),
    )
