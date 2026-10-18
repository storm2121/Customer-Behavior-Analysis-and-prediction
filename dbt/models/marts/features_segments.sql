-- Behavioural features for segmentation (M5), one row per active client. Transformations that
-- depend on the fitted population (log1p, standardisation) happen in Python; this model only
-- defines what each feature means.
select
    client_id,
    age_band,
    channel_profile,
    card_recency_days,
    n_card as card_frequency,
    card_spend as card_monetary,
    online_recency_days,
    n_online as online_frequency,
    online_amount as online_monetary,
    branch_recency_days,
    n_branch as branch_frequency,
    branch_credit_amount as branch_monetary,
    coalesce(card_share, 0) as card_share,
    coalesce(online_share, 0) as online_share,
    coalesce(branch_share, 0) as branch_share,
    coalesce(international_share, 0) as international_share,
    n_branch_deposits / nullif(active_months, 0) as deposits_per_active_month,
    coalesce(ecommerce_share, 0) as ecommerce_share,
    n_card_service_ops,
    coalesce(round_amount_share, 0) as round_amount_share,
    active_months
from {{ ref('client_360') }}
where n_events > 0
