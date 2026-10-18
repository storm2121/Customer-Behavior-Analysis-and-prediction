-- Dense daily series per channel and kind (zeros included, which count models need), with the
-- calendar columns of the models: day of week, five-day bins of the month, month, trend, bank
-- holidays and the special periods of the `calendar_periods` seed.
with days as (
    select cast(range as date) as day
    from range(
        cast({{ window_start() }} as timestamp),
        cast({{ window_end() }} as timestamp) + interval 1 day,
        interval 1 day
    )
),

series as (
    select distinct channel, kind_group as kind
    from {{ ref('int_events') }}
    where kind_group is not null
),

counts as (
    select
        event_date as day,
        channel,
        kind_group as kind,
        count(*) as n_events,
        count(distinct client_id) as n_clients,
        coalesce(sum(amount), 0) as amount
    from {{ ref('int_events') }}
    where kind_group is not null
    group by 1, 2, 3
),

periods as (
    select * from {{ ref('calendar_periods') }}
),

holidays as (
    select * from {{ ref('bank_holidays') }}
)

select
    days.day,
    series.channel,
    series.kind,
    coalesce(counts.n_events, 0) as n_events,
    coalesce(counts.n_clients, 0) as n_clients,
    coalesce(counts.amount, 0) as amount,
    isodow(days.day) as day_of_week,
    day(days.day) as day_of_month,
    case
        when day(days.day) <= 5 then '01-05'
        when day(days.day) <= 10 then '06-10'
        when day(days.day) <= 15 then '11-15'
        when day(days.day) <= 20 then '16-20'
        when day(days.day) <= 25 then '21-25'
        else '26-31'
    end as day_of_month_bin,
    month(days.day) as month,
    strftime(days.day, '%Y-%m') as year_month,
    date_diff('day', {{ window_start() }}, days.day) as trend_day,
    -- A bank holiday is a day branches close: the fixed national holidays and the two Eids, the
    -- same definition as strata.holidays.is_bank_holiday.
    (
        days.day in (select holiday_date from holidays)
        or exists (
            select 1 from periods
            where periods.period_key in ('eid_al_fitr', 'eid_al_adha')
            and days.day between periods.start_date and periods.end_date
        )
    ) as is_bank_holiday,
    exists (
        select 1 from periods
        where periods.period_key = 'ramadan' and days.day between periods.start_date and periods.end_date
    ) as is_ramadan,
    exists (
        select 1 from periods
        where periods.period_key = 'eid_al_fitr'
        and days.day between periods.start_date and periods.end_date
    ) as is_eid_al_fitr,
    exists (
        select 1 from periods
        where periods.period_key = 'eid_al_adha'
        and days.day between periods.start_date and periods.end_date
    ) as is_eid_al_adha,
    exists (
        select 1 from periods
        where periods.period_key = 'pre_eid_al_adha_week'
        and days.day between periods.start_date and periods.end_date
    ) as is_pre_eid_al_adha_week,
    not (series.channel = 'card' and days.day > {{ card_end() }}) as in_coverage
from days
cross join series
left join counts
    on counts.day = days.day and counts.channel = series.channel and counts.kind = series.kind
