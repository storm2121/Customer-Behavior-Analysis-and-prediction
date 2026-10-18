-- Per-client inputs of the monitoring layer (M6). Thresholds are project variables so that the SQL
-- and the rules engine read the same numbers. These are triage features, not accusations.
with deposits as (
    select client_id, event_date, amount
    from {{ ref('int_events') }}
    where channel = 'branch' and kind_group = 'deposit' and amount is not null
),

branch_withdrawals as (
    select client_id, event_date, amount
    from {{ ref('int_events') }}
    where channel = 'branch' and kind_group = 'withdrawal' and amount is not null
),

rolling as (
    select
        client_id,
        event_date,
        amount,
        count(*) over velocity as deposits_in_window,
        sum(amount) over velocity as deposit_sum_in_window,
        sum(case when amount < {{ var('large_deposit_threshold') }} then 1 else 0 end)
            over velocity as sub_threshold_deposits_in_window,
        sum(case when amount < {{ var('large_deposit_threshold') }} then amount else 0 end)
            over velocity as sub_threshold_sum_in_window
    from deposits
    window velocity as (
        partition by client_id
        order by event_date
        range between interval {{ var('velocity_days') - 1 }} day preceding and current row
    )
),

deposit_features as (
    select
        client_id,
        count(*) as n_branch_deposits,
        sum(amount) as branch_deposit_amount,
        max(amount) as max_branch_deposit,
        count(*) filter (where amount >= {{ var('large_deposit_threshold') }}) as n_large_deposits,
        max(deposits_in_window) as max_deposits_in_window,
        max(deposit_sum_in_window) as max_deposit_sum_in_window,
        bool_or(
            sub_threshold_deposits_in_window >= 3
            and sub_threshold_sum_in_window >= {{ var('large_deposit_threshold') }}
        ) as structuring_pattern
    from rolling
    group by client_id
),

pass_through as (
    select deposits.client_id, count(distinct deposits.event_date) as n_pass_through
    from deposits
    inner join branch_withdrawals
        on branch_withdrawals.client_id = deposits.client_id
        and branch_withdrawals.event_date > deposits.event_date
        and branch_withdrawals.event_date
            <= deposits.event_date + interval {{ var('pass_through_days') }} day
    where deposits.amount >= {{ var('pass_through_min_deposit') }}
    group by deposits.client_id
),

bursts as (
    select
        client_id,
        max(n_events) as max_month_events,
        bool_or(
            n_events >= {{ var('burst_min_events') }}
            and gap_months_before >= {{ var('burst_min_gap_months') }}
        ) as dormant_then_burst
    from {{ ref('int_client_month') }}
    group by client_id
)

select
    clients.client_id,
    clients.n_events,
    clients.card_spend,
    coalesce(deposit_features.n_branch_deposits, 0) as n_branch_deposits,
    coalesce(deposit_features.branch_deposit_amount, 0) as branch_deposit_amount,
    deposit_features.max_branch_deposit,
    coalesce(deposit_features.n_large_deposits, 0) as n_large_deposits,
    coalesce(deposit_features.max_deposits_in_window, 0) as max_deposits_in_window,
    coalesce(deposit_features.max_deposit_sum_in_window, 0) as max_deposit_sum_in_window,
    coalesce(deposit_features.structuring_pattern, false) as structuring_pattern,
    coalesce(pass_through.n_pass_through, 0) as n_pass_through,
    coalesce(deposit_features.branch_deposit_amount, 0) / nullif(clients.card_spend, 0)
        as cash_to_card_ratio,
    coalesce(bursts.max_month_events, 0) as max_month_events,
    coalesce(bursts.dormant_then_burst, false) as dormant_then_burst
from {{ ref('client_360') }} as clients
left join deposit_features using (client_id)
left join pass_through using (client_id)
left join bursts using (client_id)
