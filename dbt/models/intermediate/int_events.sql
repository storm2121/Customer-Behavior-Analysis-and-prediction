-- Every event of the three channels in one table: the grain the rest of the warehouse builds on.
-- `kind` is the finest key (card category, online service, branch operation) and `kind_group` the
-- coarse one used by the calendar and the features (branch operations grouped by category).
select
    event_id,
    client_id,
    event_date,
    'card' as channel,
    category as kind,
    category as kind_group,
    amount,
    direction,
    origin,
    is_spend,
    is_zero_amount,
    is_repeat
from {{ ref('stg_card_transactions') }}

union all

select
    event_id,
    client_id,
    event_date,
    'online' as channel,
    service as kind,
    service as kind_group,
    amount,
    direction,
    cast(null as varchar) as origin,
    false as is_spend,
    is_zero_amount,
    is_repeat
from {{ ref('stg_online_operations') }}

union all

select
    event_id,
    client_id,
    event_date,
    'branch' as channel,
    operation_key as kind,
    operation_category as kind_group,
    amount,
    direction,
    cast(null as varchar) as origin,
    false as is_spend,
    is_zero_amount,
    is_repeat
from {{ ref('stg_branch_operations') }}
