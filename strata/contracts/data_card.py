"""The data card's table of assumptions, generated from the seeds so it can never drift from them.

Every code meaning the bank did not document, every code kept without a meaning, and every
calendar date taken on trust is tagged in its seed (``assumed``). ``docs/data_card.md`` lists them
all in one table between two markers; ``python -m strata.contracts.data_card`` rewrites that table
from the seeds, and a test fails when the document is out of date.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

from strata.config import find_repo_root
from strata.contracts.vocab import seed_rows

DATA_CARD = Path("docs/data_card.md")
START = "<!-- assumptions:start (generated from dbt/seeds; do not edit by hand) -->"
END = "<!-- assumptions:end -->"


@dataclass(frozen=True, slots=True)
class Assumption:
    """One meaning taken on trust: where it lives, what it is, and why it is believed."""

    seed: str
    field: str
    code: str
    meaning: str
    status: str
    evidence: str


def _undocumented(meaning: str, column: str) -> bool:
    # A meaning that only repeats the column and code (``civilite_0``) means "no documented
    # meaning": the code is kept as a category.
    return meaning.startswith((f"{column}_", "flag_"))


def assumptions() -> list[Assumption]:
    """Every assumed or undocumented meaning in the seeds, in seed order."""
    rows = [
        Assumption(
            seed="codes_branch_operation",
            field="branch operation label",
            code=row["label"],
            meaning=(
                f"{row['operation_key']}: category {row['category']}, "
                f"direction {row['assumed_direction']}"
            ),
            status="assumed",
            evidence="rare label; meaning read from its wording",
        )
        for row in seed_rows("codes_branch_operation")
        if row["semantics"] == "assumed"
    ]
    rows.extend(
        Assumption(
            seed="codes_client",
            field=row["column_name"],
            code=row["code"],
            meaning=row["assumed_meaning"],
            status="assumed",
            evidence="usual coding of the field; not documented by the bank",
        )
        for row in seed_rows("codes_client")
        if row["evidence"] == "assumed"
    )
    rows.extend(
        Assumption(
            seed="codes_aggregate_flags",
            field=row["column_name"],
            code=row["code"],
            meaning=(
                "no documented meaning; kept as a category"
                if _undocumented(row["meaning"], row["column_name"])
                else row["meaning"]
            ),
            status=(
                "undocumented" if _undocumented(row["meaning"], row["column_name"]) else "assumed"
            ),
            evidence="aggregate file; not documented by the bank",
        )
        for row in seed_rows("codes_aggregate_flags")
        if row["evidence"] == "assumed"
    )
    rows.extend(
        Assumption(
            seed="calendar_periods",
            field="calendar period",
            code=row["period_key"],
            meaning=f"{row['label']}, {row['start_date']} to {row['end_date']}",
            status="assumed",
            evidence=row["evidence"],
        )
        for row in seed_rows("calendar_periods")
        if row["semantics"] == "assumed"
    )
    rows.extend(
        Assumption(
            seed="bank_holidays",
            field=f"bank holiday ({row['kind']})",
            code=row["holiday_date"],
            meaning=row["label"],
            status="assumed",
            evidence=row["evidence"],
        )
        for row in seed_rows("bank_holidays")
        if row["semantics"] == "assumed"
    )
    return rows


def _cell(text: str) -> str:
    return text.replace("|", "\\|")


def render_table(rows: list[Assumption] | None = None) -> str:
    """The Markdown table of :func:`assumptions`."""
    lines = [
        "| Seed | Field | Code | Meaning used | Status | Evidence |",
        "|---|---|---|---|---|---|",
    ]
    lines.extend(
        f"| `{row.seed}` | {_cell(row.field)} | `{_cell(row.code)}` | {_cell(row.meaning)} | "
        f"{row.status} | {_cell(row.evidence)} |"
        for row in (assumptions() if rows is None else rows)
    )
    return "\n".join(lines)


def render(document: str) -> str:
    """``document`` with the table between the markers replaced by the current one."""
    if START not in document or END not in document:
        msg = "the data card has no assumptions markers"
        raise ValueError(msg)
    head, rest = document.split(START, 1)
    _, tail = rest.split(END, 1)
    return f"{head}{START}\n{render_table()}\n{END}{tail}"


def update(root: Path | None = None) -> Path:
    """Rewrite the generated table of ``docs/data_card.md``."""
    path = (root or find_repo_root()) / DATA_CARD
    text = path.read_text(encoding="utf-8")
    path.write_text(render(text), encoding="utf-8", newline="\n")
    return path


if __name__ == "__main__":  # pragma: no cover
    sys.stdout.write(f"updated {update().as_posix()}\n")
