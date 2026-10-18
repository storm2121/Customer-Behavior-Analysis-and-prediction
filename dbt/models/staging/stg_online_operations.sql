-- Online operations: the online events of the canonical table, with the service key, its direction
-- and the exact-repeat flag.
with source as (
    select * from {{ source('raw', 'events') }}
    where channel = 'online'
),

services as (
    select * from {{ ref('codes_online_service') }}
)

select
    'online:' || cast(source.source_row as varchar) as event_id,
    source.client_id,
    cast(source.ts as date) as event_date,
    source.kind as service_label,
    services.service_key as service,
    nullif(services.direction, 'none') as direction,
    source.amount,
    coalesce(source.amount = 0, false) as is_zero_amount,
    row_number() over (
        partition by source.client_id, source.ts, source.kind, source.amount
        order by source.source_file, source.source_row
    ) > 1 as is_repeat,
    source.source_row
from source
left join services on services.label = source.kind
