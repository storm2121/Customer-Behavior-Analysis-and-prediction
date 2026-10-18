# ADR 0009: calendar model terms for a single year of data

Status: accepted (executing agent, M4, 2026-10-18). Deviates from ARCHITECTURE.md section 9, which
lists a trend term among the regressors of the calendar model.

## Context

ARCHITECTURE.md section 9 specifies a negative-binomial regression of daily counts on day of week,
five-day bins of the month, month, trend, Ramadan, the two Eids and the week before Eid al-Adha.
The extract covers one year: thirteen calendar months, the first and last of them partial.

With one year, month of year and a trend cannot be told apart. At the level of months a linear
trend is just one more pattern across the thirteen months, which the month terms already span.
Added next to the month terms, a trend coefficient is identified only by drift within months, and
that drift is what the day-of-month bins describe; the coefficient would then be a within-month
slope that no one would read as a trend. Separately, the warehouse marks the two Eids as bank
holidays (branches close on both), so a bank-holiday term next to the Eid terms would make each
Eid effect an effect on top of an ordinary holiday.

## Decision

- **Month, no separate trend.** Calendar month enters in sum-to-zero (deviation) coding, so each
  month's rate ratio is against the average month of the window, and the trend over the year is
  part of the month effects. Every calendar claim carries the caveat that month effects absorb any
  trend.
- **Holidays.** The bank-holiday term covers the bank holidays other than the Eids (the
  fixed-date national holidays and, since the architect supplied their Moroccan dates, the
  floating religious ones: Islamic New Year and Mawlid); the Eids have their own terms, so every
  holiday effect is read against ordinary days. The mart's `is_bank_holiday` (all bank holidays
  and the Eids, the definition of `strata.holidays`) still decides which days branches are open.
- **Branch series on open days.** Branch series are modelled on Monday to Friday outside bank
  holidays; weekend and holiday terms are not estimable there and are dropped.
- **References.** Monday for the day of week and days eleven to fifteen for the day of the month,
  so the first bin reads as the salary-day effect.

## Consequences

- "December against the average month" includes whatever trend the year had. A second year of data
  would separate the two (month of year plus a year or trend term); the design function is the
  single place to change.
- The Bayesian model (ADR 0007) uses the same design, so the two fits stay comparable.
- The method text of each calendar claim states the terms, so the published wording cannot drift
  from the model.
