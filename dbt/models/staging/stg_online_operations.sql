-- Online operations: typed, with the service key, its direction and the exact-repeat flag.
with source as (
    select * from {{ source('raw', 'online_operations') }}
),

services as (
    select * from {{ ref('codes_online_service') }}
)

select
    'online:' || cast(source._source_row as varchar) as event_id,
    source.identifiant_client as client_id,
    cast(source.operation_date_time as date) as event_date,
    source.service as service_label,
    services.service_key as service,
    nullif(services.direction, 'none') as direction,
    source.montant_transaction as amount,
    coalesce(source.montant_transaction = 0, false) as is_zero_amount,
    row_number() over (
        partition by
            source.identifiant_client,
            source.operation_date_time,
            source.service,
            source.montant_transaction
        order by source._source_row
    ) > 1 as is_repeat,
    source._source_row as source_row
from source
left join services on services.label = source.service
