"""Calendar periods that matter for Moroccan retail banking in the extract's window.

Defined once and shared: the synthetic generator uses them to shape daily intensities, and the
calendar models use them as regressors, so a period can never mean two different things.

The Islamic dates follow ARCHITECTURE.md section 9. Moroccan dates depend on the sighting of the
moon and can differ by a day from other countries; the models treat each period as a block, which
makes them robust to a one-day shift. Only fixed-date national holidays are listed; the floating
Islamic holidays other than the two Eids are left out rather than guessed.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta


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


RAMADAN_2024 = Period("ramadan", "Ramadan 1445", date(2024, 3, 11), date(2024, 4, 9))
EID_AL_FITR_2024 = Period("eid_al_fitr", "Eid al-Fitr", date(2024, 4, 10), date(2024, 4, 11))
EID_AL_ADHA_2024 = Period("eid_al_adha", "Eid al-Adha", date(2024, 6, 16), date(2024, 6, 17))
PRE_EID_AL_ADHA_WEEK_2024 = Period(
    "pre_eid_al_adha_week", "Week before Eid al-Adha", date(2024, 6, 9), date(2024, 6, 15)
)

SPECIAL_PERIODS: tuple[Period, ...] = (
    RAMADAN_2024,
    EID_AL_FITR_2024,
    EID_AL_ADHA_2024,
    PRE_EID_AL_ADHA_WEEK_2024,
)

NATIONAL_HOLIDAYS: tuple[tuple[date, str], ...] = (
    (date(2023, 11, 6), "Green March"),
    (date(2023, 11, 18), "Independence Day"),
    (date(2024, 1, 1), "New Year"),
    (date(2024, 1, 11), "Proclamation of Independence"),
    (date(2024, 1, 14), "Amazigh New Year"),
    (date(2024, 5, 1), "Labour Day"),
    (date(2024, 7, 30), "Throne Day"),
    (date(2024, 8, 14), "Oued Ed-Dahab Day"),
    (date(2024, 8, 20), "Revolution of the King and the People"),
    (date(2024, 8, 21), "Youth Day"),
)
"""Fixed-date national holidays inside the window. Branches are closed on these days."""


def is_bank_holiday(day: date) -> bool:
    """National holidays and the two Eids: days on which branches do not open."""
    return (
        any(day == holiday for holiday, _ in NATIONAL_HOLIDAYS)
        or day in EID_AL_FITR_2024
        or day in EID_AL_ADHA_2024
    )


def day_of_month_bin(day: date) -> str:
    """Five-day bins of the month (``01-05`` … ``26-31``), as used by the calendar models."""
    start = min(((day.day - 1) // 5) * 5 + 1, 26)
    end = 31 if start == 26 else start + 4
    return f"{start:02d}-{end:02d}"
