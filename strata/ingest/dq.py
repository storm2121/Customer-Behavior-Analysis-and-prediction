"""``strata dq``: data-quality measurements on the raw layer.

Each check is a pure function of the typed frames and returns counts with a sample-free description,
so the report is safe to print on a private run and every check can be unit-tested on a hand-built
frame. Status meanings: ``ok`` (nothing to report), ``info`` (a measurement), ``warning`` (a
quirk the analysis must handle, such as undecidable repeats), ``error`` (the pipeline's
assumptions do not hold;
``strata dq`` exits non-zero).

The definition of an exact repeat (same client, date, amount, label and origin) is shared with dbt
staging, which flags the rows; a dbt test reconciles its count with this report.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Literal

import duckdb
import pandas as pd

from strata.contracts.extracts import AMOUNT_COLUMNS, COLUMN_TYPES, cast_to_contract
from strata.contracts.specs import BY_NAME, WINDOW_END, WINDOW_START, Extract
from strata.contracts.vocab import (
    branch_directions,
    branch_operations,
    card_categories,
    card_origins,
    client_codes,
    online_services,
)
from strata.warehouse import read_table, replace_table, row_count, table_exists

RAW = "raw"
LANDING = "landing"

Status = Literal["ok", "info", "warning", "error"]
Frames = dict[Extract, pd.DataFrame]

REPEAT_KEYS: dict[Extract, tuple[str, ...]] = {
    Extract.CARD: (
        "identifiant_client",
        "date_transaction",
        "montant_local",
        "libelle_nature_transaction",
        "type_origine",
    ),
    Extract.ONLINE: ("identifiant_client", "operation_date_time", "service", "montant_transaction"),
    Extract.BRANCH: (
        "identifiant_client",
        "date_operation",
        "montant_operation",
        "sens_operation",
        "operation",
        "devise",
    ),
}
"""Columns that define an exact repeat. Without a transaction id or a time of day, a repeat cannot
be told apart from a genuine second transaction, so repeats are flagged and never dropped."""

IMPLAUSIBLE_CHILDREN = 15
ADULT_AGE = 18
OLDEST_PLAUSIBLE_AGE = 100


@dataclass(frozen=True, slots=True)
class DQResult:
    check_id: str
    extract: str
    status: Status
    count: int
    denominator: int | None
    description: str


Check = Callable[[Frames], list[DQResult]]
CHECKS: list[Check] = []


def check(function: Check) -> Check:
    CHECKS.append(function)
    return function


def _result(
    check_id: str,
    extract: Extract | str,
    count: int,
    denominator: int | None,
    description: str,
    *,
    when_nonzero: Status = "warning",
    when_zero: Status = "ok",
) -> DQResult:
    name = extract.value if isinstance(extract, Extract) else extract
    status = when_nonzero if count else when_zero
    return DQResult(check_id, name, status, int(count), denominator, description)


def _events(frames: Frames) -> list[Extract]:
    return [name for name in (Extract.CARD, Extract.ONLINE, Extract.BRANCH) if name in frames]


@check
def row_counts(frames: Frames) -> list[DQResult]:
    return [
        _result(
            "rows",
            name,
            len(frame),
            None,
            "rows in the file",
            when_nonzero="info",
            when_zero="error",
        )
        for name, frame in frames.items()
    ]


@check
def time_of_day(frames: Frames) -> list[DQResult]:
    results = []
    for name in _events(frames):
        dates = frames[name][BY_NAME[name].date_column or ""]
        count = int((dates != dates.dt.normalize()).sum())
        results.append(
            _result(
                "time_of_day_present",
                name,
                count,
                len(dates),
                "timestamps with a time other than midnight",
            )
        )
    return results


@check
def outside_window(frames: Frames) -> list[DQResult]:
    start, end = pd.Timestamp(WINDOW_START), pd.Timestamp(WINDOW_END)
    results = []
    for name in _events(frames):
        dates = frames[name][BY_NAME[name].date_column or ""]
        count = int(((dates < start) | (dates > end)).sum())
        results.append(
            _result("outside_window", name, count, len(dates), "events outside the rolling year")
        )
    return results


@check
def exact_repeats(frames: Frames) -> list[DQResult]:
    results = []
    for name in _events(frames):
        frame = frames[name]
        count = int(frame.duplicated(list(REPEAT_KEYS[name]), keep="first").sum())
        results.append(
            _result(
                "exact_repeats",
                name,
                count,
                len(frame),
                "rows repeating an earlier row on every field; undecidable without an id",
            )
        )
    return results


@check
def missing_values(frames: Frames) -> list[DQResult]:
    results = []
    for name in _events(frames):
        frame = frames[name]
        amount = AMOUNT_COLUMNS[name]
        results.append(
            _result(
                "missing_amount",
                name,
                int(frame[amount].isna().sum()),
                len(frame),
                "rows without an amount",
            )
        )
    if Extract.CARD in frames:
        card = frames[Extract.CARD]
        count = int(card["libelle_nature_transaction"].isna().sum())
        results.append(
            _result(
                "missing_category", Extract.CARD, count, len(card), "card rows without a category"
            )
        )
    if Extract.BRANCH in frames:
        branch = frames[Extract.BRANCH]
        count = int(branch["sens_operation"].isna().sum())
        results.append(
            _result(
                "missing_direction",
                Extract.BRANCH,
                count,
                len(branch),
                "branch rows without a direction",
            )
        )
    if Extract.CLIENTS in frames:
        clients = frames[Extract.CLIENTS]
        count = int(clients["situation_familiale"].isna().sum())
        results.append(
            _result(
                "missing_marital_status",
                Extract.CLIENTS,
                count,
                len(clients),
                "clients without a marital status",
                when_nonzero="info",
            )
        )
    return results


@check
def amounts(frames: Frames) -> list[DQResult]:
    results = []
    for name in _events(frames):
        values = frames[name][AMOUNT_COLUMNS[name]]
        results.append(
            _result(
                "zero_amount",
                name,
                int((values == 0).sum()),
                len(values),
                "rows with a zero amount",
                when_nonzero="info",
            )
        )
    if Extract.CARD in frames:
        card = frames[Extract.CARD]
        values = card["montant_local"]
        tiny = int(((values > 0) & (values < 1)).sum())
        results.append(
            _result(
                "sub_one_mad",
                Extract.CARD,
                tiny,
                len(card),
                "card amounts below one dirham",
                when_nonzero="info",
            )
        )
        withdrawals = values[card["libelle_nature_transaction"] == "WITHDRAWAL"].dropna()
        odd = int((withdrawals % 100 != 0).sum())
        results.append(
            _result(
                "withdrawal_not_multiple_of_100",
                Extract.CARD,
                odd,
                len(withdrawals),
                "withdrawals not a multiple of one hundred",
                when_nonzero="info",
            )
        )
    if Extract.ONLINE in frames:
        online = frames[Extract.ONLINE]
        zero = online["montant_transaction"] == 0
        card_service = {s.label for s in online_services() if s.zero_amount_expected}
        unexpected = int((zero & ~online["service"].isin(card_service)).sum())
        results.append(
            _result(
                "zero_amount_outside_card_service",
                Extract.ONLINE,
                unexpected,
                int(zero.sum()),
                "zero-amount online rows that are not card-service operations",
            )
        )
    return results


@check
def vocabularies(frames: Frames) -> list[DQResult]:
    def unknown(series: pd.Series, known: set[str]) -> int:
        present = series.dropna()
        return int((~present.isin(known)).sum())

    results = []
    if Extract.CARD in frames:
        card = frames[Extract.CARD]
        labels = {c.label for c in card_categories()}
        results.append(
            _result(
                "unknown_label",
                Extract.CARD,
                unknown(card["libelle_nature_transaction"], labels),
                len(card),
                "card categories outside the code map",
            )
        )
        results.append(
            _result(
                "unknown_origin",
                Extract.CARD,
                unknown(card["type_origine"], set(card_origins())),
                len(card),
                "card origins outside the code map",
            )
        )
    if Extract.ONLINE in frames:
        online = frames[Extract.ONLINE]
        labels = {s.label for s in online_services()}
        results.append(
            _result(
                "unknown_label",
                Extract.ONLINE,
                unknown(online["service"], labels),
                len(online),
                "online services outside the code map",
            )
        )
    if Extract.BRANCH in frames:
        branch = frames[Extract.BRANCH]
        labels = {op.label for op in branch_operations()}
        results.append(
            _result(
                "unknown_label",
                Extract.BRANCH,
                unknown(branch["operation"], labels),
                len(branch),
                "branch operations outside the code map",
            )
        )
        results.append(
            _result(
                "unknown_direction",
                Extract.BRANCH,
                unknown(branch["sens_operation"], set(branch_directions())),
                len(branch),
                "branch directions outside the code map",
            )
        )
        results.append(
            _result(
                "currency_not_mad",
                Extract.BRANCH,
                unknown(branch["devise"], {"MAD"}),
                len(branch),
                "branch operations in a currency other than MAD",
            )
        )
        evident = {
            op.label: op.assumed_direction for op in branch_operations() if op.semantics == "label"
        }
        expected = branch["operation"].map(evident)
        known = branch["sens_operation"].isin(set(branch_directions()))
        mismatch = int((expected.notna() & known & (expected != branch["sens_operation"])).sum())
        results.append(
            _result(
                "direction_contradicts_label",
                Extract.BRANCH,
                mismatch,
                len(branch),
                "branch rows whose direction contradicts an unambiguous label",
            )
        )
    if Extract.CLIENTS in frames:
        clients = frames[Extract.CLIENTS]
        for column in ("genre", "situation_familiale"):
            codes = set(client_codes(column))
            present = clients[column].dropna()
            count = int((~present.isin(codes)).sum())
            results.append(
                _result(
                    f"unknown_{column}",
                    Extract.CLIENTS,
                    count,
                    len(clients),
                    f"{column} codes outside the code map",
                )
            )
    return results


@check
def plausibility(frames: Frames) -> list[DQResult]:
    results = []
    if Extract.CLIENTS in frames:
        clients = frames[Extract.CLIENTS]
        children = int((clients["nombre_d_enfants"] > IMPLAUSIBLE_CHILDREN).sum())
        results.append(
            _result(
                "implausible_children",
                Extract.CLIENTS,
                children,
                len(clients),
                f"clients with more than {IMPLAUSIBLE_CHILDREN} children recorded",
            )
        )
        ages = clients["age"]
        odd = int(((ages < ADULT_AGE) | (ages > OLDEST_PLAUSIBLE_AGE)).sum())
        results.append(
            _result(
                "age_out_of_range",
                Extract.CLIENTS,
                odd,
                len(clients),
                "client ages below adulthood or above one hundred",
            )
        )
        block = ["secteur_professionnel", "profession", "categorie_socio_professionelle"]
        missing = clients[block].isna()
        results.append(
            _result(
                "profession_block_missing",
                Extract.CLIENTS,
                int(missing.all(axis=1).sum()),
                len(clients),
                "clients with no profession information",
                when_nonzero="info",
            )
        )
        partial = int((missing.any(axis=1) & ~missing.all(axis=1)).sum())
        results.append(
            _result(
                "profession_block_partial",
                Extract.CLIENTS,
                partial,
                len(clients),
                "clients with some but not all profession fields",
            )
        )
    if Extract.AGGREGATE in frames:
        aggregate = frames[Extract.AGGREGATE].dropna(subset=["identifier"])
        ages = aggregate["age"]
        odd = int(((ages < ADULT_AGE) | (ages > OLDEST_PLAUSIBLE_AGE)).sum())
        results.append(
            _result(
                "age_out_of_range",
                Extract.AGGREGATE,
                odd,
                len(aggregate),
                "aggregate-file ages below adulthood or above one hundred",
            )
        )
    return results


@check
def joins(frames: Frames) -> list[DQResult]:
    if Extract.CLIENTS not in frames:
        return []
    clients = set(frames[Extract.CLIENTS]["identifiant_client"].dropna())
    results = []
    active: set[str] = set()
    for name in _events(frames):
        ids = frames[name]["identifiant_client"]
        active |= set(ids.dropna())
        orphans = int((~ids.isin(clients)).sum())
        results.append(
            _result(
                "orphan_events",
                name,
                orphans,
                len(ids),
                "events whose client is not in the client file",
            )
        )
        coverage = int(ids[ids.isin(clients)].nunique())
        results.append(
            _result(
                "clients_with_events",
                name,
                coverage,
                len(clients),
                "clients of the client file with at least one event here",
                when_nonzero="info",
                when_zero="warning",
            )
        )
    silent = len(clients - active)
    results.append(
        _result(
            "clients_without_events",
            Extract.CLIENTS,
            silent,
            len(clients),
            "clients with no event in any channel",
            when_nonzero="info",
        )
    )
    if Extract.AGGREGATE in frames:
        identifiers = frames[Extract.AGGREGATE]["identifier"].dropna()
        matches = int(identifiers.isin(clients).sum())
        results.append(
            _result(
                "aggregate_join_coverage",
                Extract.AGGREGATE,
                matches,
                len(identifiers),
                "aggregate-file identifiers found in the client file "
                "(none means a different sample: any join is empty)",
                when_nonzero="info",
                when_zero="warning",
            )
        )
    return results


@check
def aggregate_file(frames: Frames) -> list[DQResult]:
    if Extract.AGGREGATE not in frames:
        return []
    aggregate = frames[Extract.AGGREGATE]
    data_columns = [c for c in aggregate.columns if c != "_source_row"]
    empty = int(aggregate[data_columns].isna().all(axis=1).sum())
    results = [
        _result(
            "empty_rows",
            Extract.AGGREGATE,
            empty,
            len(aggregate),
            "rows with every field empty",
            when_nonzero="info",
        )
    ]
    filled = aggregate.dropna(subset=["identifier"])
    no_online = filled["client_has_cih_online"] == 2
    contradictions = int((no_online & (filled["nombre_operations"] > 0)).sum())
    results.append(
        _result(
            "online_flag_contradicts_operations",
            Extract.AGGREGATE,
            contradictions,
            int(no_online.sum()),
            "clients coded without online access who have online operations",
        )
    )
    constant = [
        c for c in data_columns if c != "identifier" and filled[c].nunique(dropna=False) <= 1
    ]
    results.append(
        _result(
            "constant_columns",
            Extract.AGGREGATE,
            len(constant),
            len(data_columns),
            "columns holding a single value for every client",
            when_nonzero="info",
        )
    )
    return results


def run_checks(frames: Frames) -> list[DQResult]:
    """Every registered check, in registration order."""
    return [result for function in CHECKS for result in function(frames)]


def typed_raw_frames(con: duckdb.DuckDBPyConnection) -> Frames:
    """Read the landing tables back with the contract's dtypes (DuckDB's defaults differ).

    Only files read through a typed contract of the reference extract have a landing table; the
    source-specific checks above run on them.
    """
    frames: Frames = {}
    for name, types in COLUMN_TYPES.items():
        if not table_exists(con, LANDING, name.value):
            continue
        stored = read_table(con, LANDING, name.value).sort_values("_source_row", kind="stable")
        typed = {column: cast_to_contract(stored[column], kind) for column, kind in types.items()}
        frames[name] = pd.DataFrame(typed).reset_index(drop=True)
    return frames


CANONICAL_REPEAT_KEYS: dict[str, tuple[str, ...]] = {
    "card": ("client_id", "ts", "amount", "kind", "origin"),
    "online": ("client_id", "ts", "kind", "amount"),
    "branch": ("client_id", "ts", "amount", "direction", "kind", "currency"),
}
"""The exact-repeat definition on canonical columns: the same fields as :data:`REPEAT_KEYS`."""

CHANNEL_EXTRACTS: dict[str, Extract] = {
    "card": Extract.CARD,
    "online": Extract.ONLINE,
    "branch": Extract.BRANCH,
}


def canonical_results(con: duckdb.DuckDBPyConnection, covered: set[Extract]) -> list[DQResult]:
    """Source-agnostic checks on the canonical tables, for channels without a landing table.

    A mapped source without a reference contract gets the checks every source can answer: exact
    repeats (reported under the reference extract's name of the channel, so dbt's reconciliation
    of the repeat flags works for every source), events outside the window, zero amounts, and
    events whose client is not in the client file.
    """
    if not table_exists(con, RAW, "events"):
        return []
    events = read_table(con, RAW, "events").sort_values(["source_file", "source_row"])
    clients = read_table(con, RAW, "clients")
    start, end = pd.Timestamp(WINDOW_START), pd.Timestamp(WINDOW_END)
    results: list[DQResult] = []
    for channel, extract in CHANNEL_EXTRACTS.items():
        if extract in covered:
            continue
        rows = events[events["channel"] == channel]
        repeats = int(rows.duplicated(list(CANONICAL_REPEAT_KEYS[channel]), keep="first").sum())
        dates = pd.to_datetime(rows["ts"])
        results.extend(
            [
                _result(
                    "exact_repeats",
                    extract,
                    repeats,
                    len(rows),
                    "rows repeating an earlier row on every field; undecidable without an id",
                ),
                _result(
                    "outside_window",
                    extract,
                    int(((dates < start) | (dates > end)).sum()),
                    len(rows),
                    "events outside the rolling year",
                ),
                _result(
                    "zero_amounts",
                    extract,
                    int((rows["amount"] == 0).sum()),
                    len(rows),
                    "events with an amount of exactly zero",
                    when_nonzero="info",
                ),
            ]
        )
    if len(covered & set(CHANNEL_EXTRACTS.values())) < len(CHANNEL_EXTRACTS):
        unknown = int((~events["client_id"].isin(clients["client_id"])).sum())
        results.append(
            _result(
                "events_without_client",
                "events",
                unknown,
                len(events),
                "events whose client is not in the client file",
            )
        )
    return results


def _stored_rows(con: duckdb.DuckDBPyConnection, row: object) -> int:
    """Rows of one manifest entry in the warehouse: its landing table, or its canonical rows."""
    # A warehouse loaded before mappings existed has no role column; its files all have landing
    # tables (or raw tables of the same name), so the role is only needed for mapped sources.
    extract, role, filename = (
        str(getattr(row, name, "")) for name in ("extract", "role", "filename")
    )
    if table_exists(con, LANDING, extract):
        return row_count(con, LANDING, extract)
    if role == "events":
        found = con.execute(
            "select count(*) from raw.events where source_file = ?", [filename]
        ).fetchone()
        return int(found[0]) if found else 0
    return row_count(con, RAW, role)


def reconciliation_results(con: duckdb.DuckDBPyConnection) -> list[DQResult]:
    """Row counts in the warehouse against the manifest of the load."""
    if not table_exists(con, RAW, "manifest"):
        return [
            _result(
                "manifest_present",
                "warehouse",
                1,
                None,
                "no manifest table; run strata ingest",
                when_nonzero="error",
            )
        ]
    manifest = read_table(con, RAW, "manifest")
    results = []
    for row in manifest.itertuples(index=False):
        gap = abs(_stored_rows(con, row) - int(str(row.rows)))
        results.append(
            _result(
                "rows_match_manifest",
                str(row.extract),
                gap,
                int(str(row.rows)),
                "row-count gap between warehouse and manifest",
                when_nonzero="error",
            )
        )
    return results


def run(con: duckdb.DuckDBPyConnection) -> list[DQResult]:
    """Run every check against the warehouse and store the report in ``raw.dq_report``."""
    frames = typed_raw_frames(con)
    results = reconciliation_results(con) + run_checks(frames) + canonical_results(con, set(frames))
    replace_table(con, RAW, "dq_report", pd.DataFrame([asdict(r) for r in results]))
    return results


def has_errors(results: list[DQResult]) -> bool:
    return any(result.status == "error" for result in results)
