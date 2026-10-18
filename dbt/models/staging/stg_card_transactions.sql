-- Card transactions: the card events of the canonical table, with category and origin keys and the
-- exact-repeat flag. A repeat matches an earlier row on client, timestamp, amount, kind and origin.
-- With no transaction id and no time of day it cannot be told from a double posting, so it is
-- flagged, never dropped. The same definition is used by `strata dq`; a singular test reconciles
-- the two counts.
with source as (
    select * from {{ source('raw', 'events') }}
    where channel = 'card'
),

categories as (
    select * from {{ ref('codes_card_category') }}
),

origins as (
    select * from {{ ref('codes_card_origin') }}
)

select
    'card:' || cast(source.source_row as varchar) as event_id,
    source.client_id,
    cast(source.ts as date) as event_date,
    source.amount,
    source.kind as category_label,
    categories.category_key as category,
    categories.direction,
    coalesce(categories.is_spend, false) as is_spend,
    source.origin as origin_label,
    origins.origin_key as origin,
    coalesce(source.amount = 0, false) as is_zero_amount,
    row_number() over (
        partition by source.client_id, source.ts, source.amount, source.kind, source.origin
        order by source.source_file, source.source_row
    ) > 1 as is_repeat,
    source.source_row
from source
left join categories on categories.label = source.kind
left join origins on origins.label = source.origin
