-- Silence features for every cutoff of the `silence_cutoffs` seed (ADR 0003): features from events
-- on or before the cutoff only, the label from the outcome window that follows it. The universe of
-- a cutoff is every client with at least one event in the observation window.
with cutoffs as (
    select * from {{ ref('silence_cutoffs') }}
),

events as (
    select client_id, event_date, channel, amount, is_spend, direction
    from {{ ref('int_events') }}
),

observed as (
    select
        cutoffs.cutoff_date,
        events.client_id,
        count(*) as n_events,
        count(*) filter (where events.channel = 'card') as n_card,
        count(*) filter (where events.channel = 'online') as n_online,
        count(*) filter (where events.channel = 'branch') as n_branch,
        count(*) filter (
            where events.event_date > cutoffs.cutoff_date - interval 30 day
        ) as n_events_30d,
        count(*) filter (
            where events.event_date > cutoffs.cutoff_date - interval 90 day
        ) as n_events_90d,
        count(distinct date_trunc('month', events.event_date)) as active_months,
        min(events.event_date) as first_event_date,
        max(events.event_date) as last_event_date,
        coalesce(
            sum(events.amount) filter (where events.channel = 'card' and events.is_spend), 0
        ) as card_spend,
        coalesce(sum(events.amount) filter (where events.channel = 'online'), 0) as online_amount,
        coalesce(
            sum(events.amount) filter (
                where events.channel = 'branch' and events.direction = 'credit'
            ),
            0
        ) as branch_credit
    from cutoffs
    inner join events
        on events.event_date >= {{ window_start() }}
        and events.event_date <= cutoffs.cutoff_date
    group by 1, 2
),

outcome as (
    select cutoffs.cutoff_date, events.client_id, count(*) as n_events_outcome
    from cutoffs
    inner join events
        on events.event_date >= cutoffs.outcome_start
        and events.event_date <= cutoffs.outcome_end
    group by 1, 2
)

select
    observed.cutoff_date,
    cutoffs.outcome_start,
    cutoffs.outcome_end,
    cutoffs.role,
    observed.client_id,
    observed.n_events,
    observed.n_card,
    observed.n_online,
    observed.n_branch,
    observed.n_events_30d,
    observed.n_events_90d,
    observed.active_months,
    observed.first_event_date,
    observed.last_event_date,
    date_diff('day', observed.last_event_date, observed.cutoff_date) as recency_days,
    date_diff('day', observed.first_event_date, observed.cutoff_date) as active_span_days,
    observed.card_spend,
    observed.online_amount,
    observed.branch_credit,
    {{ channel_profile('observed.n_card', 'observed.n_online', 'observed.n_branch') }}
        as channel_profile,
    clients.age,
    clients.age_band,
    clients.tenure_years,
    clients.tenure_band,
    clients.gender_code,
    clients.marital_code,
    clients.children,
    clients.profession_missing,
    coalesce(outcome.n_events_outcome, 0) as n_events_outcome,
    coalesce(outcome.n_events_outcome, 0) = 0 as silent
from observed
inner join cutoffs using (cutoff_date)
inner join {{ ref('stg_clients') }} as clients using (client_id)
left join outcome using (cutoff_date, client_id)
