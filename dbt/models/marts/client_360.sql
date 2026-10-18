-- One row per client of the sample: demographics, the whole-year channel profile, per-channel
-- recency, frequency and monetary value, and the shares the segments and the ledger use.
with per_client as (
    select
        client_id,
        count(*) as n_events,
        count(*) filter (where channel = 'card') as n_card,
        count(*) filter (where channel = 'online') as n_online,
        count(*) filter (where channel = 'branch') as n_branch,
        count(*) filter (where is_repeat) as n_repeat_rows,
        count(distinct date_trunc('month', event_date)) as active_months,
        min(event_date) as first_event_date,
        max(event_date) as last_event_date,
        max(event_date) filter (where channel = 'card') as last_card_date,
        max(event_date) filter (where channel = 'online') as last_online_date,
        max(event_date) filter (where channel = 'branch') as last_branch_date,
        count(*) filter (where channel = 'card' and is_spend) as n_card_spend,
        coalesce(sum(amount) filter (where channel = 'card' and is_spend), 0) as card_spend,
        count(*) filter (where channel = 'card' and kind = 'ecommerce') as n_ecommerce,
        count(*) filter (where channel = 'card' and origin = 'international') as n_card_international,
        count(*) filter (where channel = 'card' and kind = 'withdrawal') as n_withdrawals,
        coalesce(
            sum(amount) filter (where channel = 'card' and kind = 'withdrawal'), 0
        ) as withdrawal_amount,
        count(*) filter (where channel = 'card' and kind = 'deposit') as n_card_deposits,
        count(*) filter (
            where channel = 'card' and amount is not null and amount % 100 = 0
        ) as n_card_round_amounts,
        coalesce(sum(amount) filter (where channel = 'online'), 0) as online_amount,
        count(*) filter (where channel = 'online' and kind = 'card_service') as n_card_service_ops,
        count(*) filter (where channel = 'branch' and kind_group = 'deposit') as n_branch_deposits,
        coalesce(
            sum(amount) filter (where channel = 'branch' and direction = 'credit'), 0
        ) as branch_credit_amount,
        coalesce(
            sum(amount) filter (where channel = 'branch' and direction = 'debit'), 0
        ) as branch_debit_amount
    from {{ ref('int_events') }}
    group by client_id
)

select
    clients.client_id,
    clients.age,
    clients.age_band,
    clients.tenure_years,
    clients.tenure_band,
    clients.gender_code,
    clients.gender,
    clients.marital_code,
    clients.marital_status,
    clients.children,
    clients.profession_missing,
    {{ channel_profile('per_client.n_card', 'per_client.n_online', 'per_client.n_branch') }}
        as channel_profile,
    coalesce(per_client.n_events, 0) as n_events,
    coalesce(per_client.n_card, 0) as n_card,
    coalesce(per_client.n_online, 0) as n_online,
    coalesce(per_client.n_branch, 0) as n_branch,
    coalesce(per_client.n_repeat_rows, 0) as n_repeat_rows,
    coalesce(per_client.active_months, 0) as active_months,
    per_client.first_event_date,
    per_client.last_event_date,
    date_diff('day', per_client.last_event_date, {{ window_end() }}) as recency_days,
    date_diff('day', per_client.last_card_date, {{ window_end() }}) as card_recency_days,
    date_diff('day', per_client.last_online_date, {{ window_end() }}) as online_recency_days,
    date_diff('day', per_client.last_branch_date, {{ window_end() }}) as branch_recency_days,
    coalesce(per_client.n_card_spend, 0) as n_card_spend,
    coalesce(per_client.card_spend, 0) as card_spend,
    coalesce(per_client.n_ecommerce, 0) as n_ecommerce,
    coalesce(per_client.n_card_international, 0) as n_card_international,
    coalesce(per_client.n_withdrawals, 0) as n_withdrawals,
    coalesce(per_client.withdrawal_amount, 0) as withdrawal_amount,
    coalesce(per_client.n_card_deposits, 0) as n_card_deposits,
    coalesce(per_client.online_amount, 0) as online_amount,
    coalesce(per_client.n_card_service_ops, 0) as n_card_service_ops,
    coalesce(per_client.n_branch_deposits, 0) as n_branch_deposits,
    coalesce(per_client.branch_credit_amount, 0) as branch_credit_amount,
    coalesce(per_client.branch_debit_amount, 0) as branch_debit_amount,
    per_client.n_card / nullif(per_client.n_events, 0) as card_share,
    per_client.n_online / nullif(per_client.n_events, 0) as online_share,
    per_client.n_branch / nullif(per_client.n_events, 0) as branch_share,
    per_client.n_card_international / nullif(per_client.n_card, 0) as international_share,
    per_client.n_ecommerce / nullif(per_client.n_card_spend, 0) as ecommerce_share,
    per_client.n_card_round_amounts / nullif(per_client.n_card, 0) as round_amount_share
from {{ ref('stg_clients') }} as clients
left join per_client using (client_id)
