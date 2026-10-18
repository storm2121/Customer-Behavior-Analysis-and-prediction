"""Contracts: what the pipeline relies on, enforced with counts and never with values."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from strata.contracts.extracts import (
    COLUMN_TYPES,
    SCHEMAS,
    ColumnType,
    ContractError,
    Violation,
    validate,
)
from strata.contracts.specs import AGGREGATE, CARD, CLIENTS, EXTRACTS, Extract, spec_for
from strata.ingest.readers import parse_column, parse_frame, read_extract
from strata.synthetic.parametric import SyntheticExtract

SECRET = "syn_" + "beef" * 3


def test_every_extract_has_a_type_for_every_column() -> None:
    for spec in EXTRACTS:
        assert list(COLUMN_TYPES[spec.name]) == list(spec.columns)
        assert list(SCHEMAS[spec.name].columns) == list(spec.columns)


def test_specs_are_looked_up_by_logical_name() -> None:
    assert spec_for("card_transactions") is CARD
    assert CARD.raw_table == "raw.card_transactions"
    assert CARD.event
    assert not CLIENTS.event


def test_generated_frames_satisfy_their_contracts(synthetic_extract: SyntheticExtract) -> None:
    for spec in EXTRACTS:
        validate(synthetic_extract.frames[spec.name], spec)


def test_a_negative_amount_is_a_violation_counted_without_values(
    synthetic_extract: SyntheticExtract,
) -> None:
    card = synthetic_extract.frames[Extract.CARD].copy()
    card.loc[card.index[:3], "montant_local"] = -1234.5
    with pytest.raises(ContractError) as error:
        validate(card, CARD)
    assert error.value.violations == [
        Violation("card_transactions", "montant_local", "amount_non_negative", 3)
    ]
    assert "1234" not in str(error.value)


def test_duplicate_identifiers_are_reported_without_the_identifier(
    synthetic_extract: SyntheticExtract,
) -> None:
    clients = synthetic_extract.frames[Extract.CLIENTS].copy()
    clients.loc[clients.index[:2], "identifiant_client"] = SECRET
    with pytest.raises(ContractError) as error:
        validate(clients, CLIENTS)
    assert SECRET not in str(error.value)
    assert {v.column for v in error.value.violations} == {"identifiant_client"}


def test_aggregate_identifiers_may_be_empty_but_not_repeated(
    synthetic_extract: SyntheticExtract,
) -> None:
    aggregate = synthetic_extract.frames[Extract.AGGREGATE].copy()
    empty = pd.DataFrame(
        {column: pd.Series([None], dtype=aggregate[column].dtype) for column in AGGREGATE.columns}
    )
    padded = pd.concat([empty, aggregate, empty], ignore_index=True)
    validate(padded, AGGREGATE)
    padded.loc[1, "identifier"] = padded.loc[2, "identifier"]
    with pytest.raises(ContractError) as error:
        validate(padded, AGGREGATE)
    assert error.value.violations[0].check == "unique_when_present"


def test_event_rows_need_an_identifier_and_a_date(synthetic_extract: SyntheticExtract) -> None:
    card = synthetic_extract.frames[Extract.CARD].copy()
    card.loc[card.index[0], "identifiant_client"] = pd.NA
    card.loc[card.index[1], "date_transaction"] = pd.NaT
    with pytest.raises(ContractError) as error:
        validate(card, CARD)
    assert {v.column for v in error.value.violations} == {"identifiant_client", "date_transaction"}


def test_columns_must_match_the_contract_exactly(synthetic_extract: SyntheticExtract) -> None:
    card = synthetic_extract.frames[Extract.CARD]
    with pytest.raises(ContractError):
        validate(card.assign(extra=1), CARD)
    with pytest.raises(ContractError):
        validate(card[list(reversed(CARD.columns))], CARD)


@pytest.mark.parametrize(
    ("values", "kind", "failures", "expected"),
    [
        (["1", "", "3"], ColumnType.INT, 0, [1, None, 3]),
        (["1", "x", "2.5"], ColumnType.INT, 2, [1, None, None]),
        (["1.25", "", "abc"], ColumnType.FLOAT, 1, [1.25, None, None]),
        (["a", "", "b"], ColumnType.TEXT, 0, ["a", None, "b"]),
    ],
)
def test_columns_are_parsed_explicitly(
    values: list[str], kind: ColumnType, failures: int, expected: list[object]
) -> None:
    typed, failed = parse_column(pd.Series(values, dtype="str"), kind)
    assert failed == failures
    assert [None if pd.isna(v) else v for v in typed] == expected


def test_dates_must_use_the_extract_format() -> None:
    typed, failed = parse_column(
        pd.Series(["2024-01-02T00:00", "02/01/2024", ""], dtype="str"), ColumnType.DATE
    )
    assert failed == 1
    assert typed.iloc[0] == pd.Timestamp("2024-01-02")
    assert typed.dtype == np.dtype("datetime64[us]")


def test_parse_frame_reports_every_bad_column() -> None:
    raw = pd.DataFrame(
        {
            "identifiant_client": ["a", "b"],
            "date_transaction": ["2024-01-01T00:00", "nope"],
            "montant_local": ["10", "ten"],
            "libelle_nature_transaction": ["WITHDRAWAL", ""],
            "type_origine": ["National", "National"],
        },
        dtype="str",
    )
    with pytest.raises(ContractError) as error:
        parse_frame(raw, CARD)
    assert {(v.column, v.check) for v in error.value.violations} == {
        ("date_transaction", "parse_date"),
        ("montant_local", "parse_float"),
    }


def write(path: Path, text: str) -> Path:
    path.write_bytes(text.encode("latin-1"))
    return path


def test_read_extract_rejects_a_wrong_header(tmp_path: Path) -> None:
    path = write(tmp_path / CARD.filename, "a,b,c\r\n1,2,3\r\n")
    with pytest.raises(ContractError, match="columns_match_spec"):
        read_extract(path, CARD)


def test_read_extract_counts_lines_independently_of_the_parser(tmp_path: Path) -> None:
    header = ",".join(CARD.columns)
    text = f'{header}\r\nx,2024-01-01T00:00,1,"WITH\r\nDRAWAL",National\r\n'
    with pytest.raises(ContractError, match="rows_equal_lines"):
        read_extract(write(tmp_path / CARD.filename, text), CARD)


def test_read_extract_reports_file_facts(tmp_path: Path) -> None:
    header = ",".join(CARD.columns)
    text = f"{header}\r\nx,2024-01-01T00:00,1600,WITHDRAWAL,National\r\n"
    result = read_extract(write(tmp_path / CARD.filename, text), CARD)
    assert result.data_lines == 1
    assert result.line_terminator == "CRLF"
    assert result.frame["montant_local"].iloc[0] == 1600.0
    lf = read_extract(write(tmp_path / "lf.csv", text.replace("\r\n", "\n")), CARD)
    assert lf.line_terminator == "LF"
