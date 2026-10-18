-- Card transactions: typed, with category and origin keys and the exact-repeat flag.
-- A repeat matches an earlier row on client, date, amount, label and origin. With no transaction id
-- and no time of day it cannot be told from a double posting, so it is flagged, never dropped. The
-- same definition is used by `strata dq`; a singular test reconciles the two counts.
with source as (
    select * from {{ source('raw', 'card_transactions') }}
),

categories as (
    select * from {{ ref('codes_card_category') }}
),

origins as (
    select * from {{ ref('codes_card_origin') }}
)

select
    'card:' || cast(source._source_row as varchar) as event_id,
    source.identifiant_client as client_id,
    cast(source.date_transaction as date) as event_date,
    source.montant_local as amount,
    source.libelle_nature_transaction as category_label,
    categories.category_key as category,
    categories.direction,
    coalesce(categories.is_spend, false) as is_spend,
    source.type_origine as origin_label,
    origins.origin_key as origin,
    coalesce(source.montant_local = 0, false) as is_zero_amount,
    row_number() over (
        partition by
            source.identifiant_client,
            source.date_transaction,
            source.montant_local,
            source.libelle_nature_transaction,
            source.type_origine
        order by source._source_row
    ) > 1 as is_repeat,
    source._source_row as source_row
from source
left join categories on categories.label = source.libelle_nature_transaction
left join origins on origins.label = source.type_origine
