"""Calendar periods that matter for Moroccan retail banking in the extract's window.

Defined once, in the dbt seeds ``calendar_periods.csv`` and ``bank_holidays.csv``, and shared: dbt
builds the calendar columns of ``daily_counts`` from them, the synthetic generator shapes daily
intensities with them and the calendar models use them as regressors, so a period can never mean
two different things.

The Islamic dates follow ARCHITECTURE.md section 9. Moroccan dates depend on the sighting of the
moon and can differ by a day from other countries; the models treat each period as a block, which
makes them robust to a one-day shift. Only fixed-date national holidays are listed; the floating
Islamic holidays other than the two Eids are left out rather than guessed.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

from kashf.contracts.vocab import calendar_rows, holiday_rows


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

NATIONAL_HOLIDAYS: tuple[tuple[date, str], ...] = holiday_rows()
"""Fixed-date national holidays inside the window. Branches are closed on these days."""


def is_bank_holiday(day: date) -> bool:
    """National holidays and the two Eids: days on which branches do not open."""
    return (
        any(day == holiday for holiday, _ in NATIONAL_HOLIDAYS)
        or day in EID_AL_FITR_2024
        or day in EID_AL_ADHA_2024
    )
