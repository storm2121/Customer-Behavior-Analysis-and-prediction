-- Branch operations: the branch events of the canonical table, with the operation key and category,
-- the direction and the repeat flag. The direction comes from the source (the reference extract's
-- `sens_operation`); the seed's assumed direction is only used by the data-quality check that
-- compares the two.
with source as (
    select * from {{ source('raw', 'events') }}
    where channel = 'branch'
),

operations as (
    select * from {{ ref('codes_branch_operation') }}
),

directions as (
    select * from {{ ref('codes_branch_direction') }}
)

select
    'branch:' || cast(source.source_row as varchar) as event_id,
    source.client_id,
    cast(source.ts as date) as event_date,
    source.amount,
    source.direction as direction_code,
    directions.direction,
    source.kind as operation_label,
    operations.operation_key,
    operations.category as operation_category,
    coalesce(operations.is_cash_deposit, false) as is_cash_deposit,
    source.currency,
    coalesce(source.amount = 0, false) as is_zero_amount,
    row_number() over (
        partition by
            source.client_id,
            source.ts,
            source.amount,
            source.direction,
            source.kind,
            source.currency
        order by source.source_file, source.source_row
    ) > 1 as is_repeat,
    source.source_row
from source
left join operations on operations.label = source.kind
left join directions on directions.code = source.direction
