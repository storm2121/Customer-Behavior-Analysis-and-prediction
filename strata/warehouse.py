"""DuckDB access.

The only SQL written in Python lives here and is data definition with identifiers that come from
code constants (validated below), never from data. Transformations belong to dbt.
"""

from __future__ import annotations

import re
from pathlib import Path

import duckdb
import pandas as pd

_IDENTIFIER = re.compile(r"^[a-z_][a-z0-9_]*$")


def _identifier(name: str) -> str:
    if not _IDENTIFIER.match(name):
        msg = f"not a safe SQL identifier: {name!r}"
        raise ValueError(msg)
    return name


def connect(path: Path, *, read_only: bool = False) -> duckdb.DuckDBPyConnection:
    """Open the warehouse file, creating its folder when writing."""
    if not read_only:
        path.parent.mkdir(parents=True, exist_ok=True)
    return duckdb.connect(str(path), read_only=read_only)


def replace_table(
    con: duckdb.DuckDBPyConnection, schema: str, table: str, frame: pd.DataFrame
) -> None:
    """Create or replace ``schema.table`` from a frame, in one statement."""
    schema, table = _identifier(schema), _identifier(table)
    con.execute(f"create schema if not exists {schema}")
    view = f"_strata_frame_{table}"
    con.register(view, frame)
    try:
        con.execute(f"create or replace table {schema}.{table} as select * from {view}")  # noqa: S608
    finally:
        con.unregister(view)


def read_table(con: duckdb.DuckDBPyConnection, schema: str, table: str) -> pd.DataFrame:
    """A whole table or view as a frame, in a canonical row order.

    DuckDB scans in parallel, so without an order the same table can come back in a different
    order after a rebuild. Order decides which bootstrap weight lands on which client, so every
    read sorts on all columns: the same data always gives the same frame and the same intervals.
    """
    relation = f"{_identifier(schema)}.{_identifier(table)}"
    return con.sql(f"select * from {relation} order by all").df()  # noqa: S608 - validated


def read_relation(con: duckdb.DuckDBPyConnection, name: str) -> pd.DataFrame:
    """``read_table`` for a qualified ``schema.table`` name."""
    schema, _, table = name.partition(".")
    return read_table(con, schema, table)


def table_exists(con: duckdb.DuckDBPyConnection, schema: str, table: str) -> bool:
    found = con.execute(
        "select count(*) from information_schema.tables where table_schema = ? and table_name = ?",
        [schema, table],
    ).fetchone()
    return bool(found and found[0])


def row_count(con: duckdb.DuckDBPyConnection, schema: str, table: str) -> int:
    """Rows of a table or a view (identifiers are validated, so the query cannot be injected)."""
    query = f"select count(*) from {_identifier(schema)}.{_identifier(table)}"  # noqa: S608
    found = con.execute(query).fetchone()
    return int(found[0]) if found else 0
