"""Calendar periods that matter for Moroccan retail banking in the extract's window.

Defined once, in the dbt seeds ``calendar_periods.csv`` and ``bank_holidays.csv``, and shared: dbt
builds the calendar columns of ``daily_counts`` from them, the synthetic generator shapes daily
intensities with them and the calendar models use them as regressors, so a period can never mean
two different things.

Moroccan Islamic dates follow the official sighting of the moon, which usually comes a day after
Saudi Arabia's; the seeds carry the Moroccan dates given by the architect, tagged ``assumed`` until
Mohamed confirms them. The models treat each period as a block, which keeps them robust to a
one-day shift. Bank holidays are the fixed-date national holidays, the floating religious holidays
(Islamic New Year, Mawlid) and the two Eids.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

from strata.contracts.vocab import calendar_rows, holiday_rows


@dataclass(frozen=True, slots=True)
class Period:
    """An inclusive date range with a stable key used in column names."""

    key: str
    label: str
    start: date
    end: date

    def __post_init__(self) -> None:
        if self.end < self.start:
            msg = f"period {self.key} ends before it starts"
            raise ValueError(msg)

    def __contains__(self, day: object) -> bool:
        return isinstance(day, date) and self.start <= day <= self.end

    def days(self) -> list[date]:
        return [
            self.start + timedelta(days=offset)
            for offset in range((self.end - self.start).days + 1)
        ]


_PERIODS: dict[str, Period] = {
    key: Period(key, label, start, end) for key, label, start, end in calendar_rows()
}

RAMADAN_2024 = _PERIODS["ramadan"]
EID_AL_FITR_2024 = _PERIODS["eid_al_fitr"]
EID_AL_ADHA_2024 = _PERIODS["eid_al_adha"]
PRE_EID_AL_ADHA_WEEK_2024 = _PERIODS["pre_eid_al_adha_week"]

SPECIAL_PERIODS: tuple[Period, ...] = tuple(_PERIODS.values())

BANK_HOLIDAYS: tuple[tuple[date, str], ...] = holiday_rows()
"""Bank holidays inside the window other than the two Eids: fixed national and floating religious
holidays. Branches are closed on these days."""


def is_bank_holiday(day: date) -> bool:
    """Bank holidays and the two Eids: days on which branches do not open."""
    return (
        any(day == holiday for holiday, _ in BANK_HOLIDAYS)
        or day in EID_AL_FITR_2024
        or day in EID_AL_ADHA_2024
    )
