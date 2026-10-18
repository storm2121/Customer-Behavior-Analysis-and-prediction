-- Branch operations: typed, with the operation key and category, the direction and the repeat flag.
-- The direction comes from the extract (`sens_operation`); the seed's assumed direction is only
-- used by the data-quality check that compares the two.
with source as (
    select * from {{ source('raw', 'branch_operations') }}
),

operations as (
    select * from {{ ref('codes_branch_operation') }}
),

directions as (
    select * from {{ ref('codes_branch_direction') }}
)

select
    'branch:' || cast(source._source_row as varchar) as event_id,
    source.identifiant_client as client_id,
    cast(source.date_operation as date) as event_date,
    source.montant_operation as amount,
    source.sens_operation as direction_code,
    directions.direction,
    source.operation as operation_label,
    operations.operation_key,
    operations.category as operation_category,
    coalesce(operations.is_cash_deposit, false) as is_cash_deposit,
    source.devise as currency,
    coalesce(source.montant_operation = 0, false) as is_zero_amount,
    row_number() over (
        partition by
            source.identifiant_client,
            source.date_operation,
            source.montant_operation,
            source.sens_operation,
            source.operation,
            source.devise
        order by source._source_row
    ) > 1 as is_repeat,
    source._source_row as source_row
from source
left join operations on operations.label = source.operation
left join directions on directions.code = source.sens_operation
