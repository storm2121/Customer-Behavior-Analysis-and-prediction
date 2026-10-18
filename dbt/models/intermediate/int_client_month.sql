-- A dense client-month panel: every client in every month of the window, active or not, with
-- per-channel counts and amounts and window functions for rolling sums and recency.
with months as (
    select
        cast(range as date) as month_start,
        cast(range + interval 1 month - interval 1 day as date) as month_end
    from range(
        date_trunc('month', {{ window_start() }}),
        date_trunc('month', {{ window_end() }}) + interval 1 month,
        interval 1 month
    )
),

panel as (
    select clients.client_id, months.month_start, months.month_end
    from {{ ref('stg_clients') }} as clients
    cross join months
),

activity as (
    select
        client_id,
        cast(date_trunc('month', event_date) as date) as month_start,
        count(*) filter (where channel = 'card') as n_card,
        count(*) filter (where channel = 'online') as n_online,
        count(*) filter (where channel = 'branch') as n_branch,
        count(*) as n_events,
        coalesce(sum(amount) filter (where channel = 'card' and is_spend), 0) as card_spend,
        coalesce(sum(amount) filter (where channel = 'online'), 0) as online_amount,
        coalesce(
            sum(amount) filter (where channel = 'branch' and direction = 'credit'), 0
        ) as branch_credit,
        coalesce(
            sum(amount) filter (where channel = 'branch' and direction = 'debit'), 0
        ) as branch_debit
    from {{ ref('int_events') }}
    group by 1, 2
),

joined as (
    select
        panel.client_id,
        panel.month_start,
        date_diff(
            'day',
            greatest(panel.month_start, {{ window_start() }}),
            least(panel.month_end, {{ window_end() }})
        ) + 1 as days_covered,
        coalesce(activity.n_card, 0) as n_card,
        coalesce(activity.n_online, 0) as n_online,
        coalesce(activity.n_branch, 0) as n_branch,
        coalesce(activity.n_events, 0) as n_events,
        coalesce(activity.card_spend, 0) as card_spend,
        coalesce(activity.online_amount, 0) as online_amount,
        coalesce(activity.branch_credit, 0) as branch_credit,
        coalesce(activity.branch_debit, 0) as branch_debit
    from panel
    left join activity using (client_id, month_start)
),

windowed as (
    select
        *,
        days_covered < 28 as is_partial_month,
        sum(n_events) over rolling_3 as events_rolling_3m,
        sum(card_spend) over rolling_3 as card_spend_rolling_3m,
        sum(n_events) over to_date as events_to_date,
        max(case when n_events > 0 then month_start end) over to_date as last_active_month,
        max(case when n_events > 0 then month_start end) over before_now as previous_active_month
    from joined
    window
        rolling_3 as (
            partition by client_id order by month_start rows between 2 preceding and current row
        ),
        to_date as (
            partition by client_id order by month_start rows between unbounded preceding and current row
        ),
        before_now as (
            partition by client_id order by month_start rows between unbounded preceding and 1 preceding
        )
)

select
    *,
    date_diff('month', last_active_month, month_start) as months_since_active,
    date_diff('month', previous_active_month, month_start) as gap_months_before
from windowed
