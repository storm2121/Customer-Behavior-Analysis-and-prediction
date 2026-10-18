-- Row counts of the warehouse against the ingest manifest: every staged event file keeps all its
-- rows, the event table is their union, and the client marts hold every client exactly once.
with manifest as (
    select extract, rows from {{ source('raw', 'manifest') }}
),

expected as (
    select 'stg_clients' as model, rows from manifest where extract = 'clients'
    union all
    select 'client_360', rows from manifest where extract = 'clients'
    union all
    select 'stg_card_transactions', rows from manifest where extract = 'card_transactions'
    union all
    select 'stg_online_operations', rows from manifest where extract = 'online_operations'
    union all
    select 'stg_branch_operations', rows from manifest where extract = 'branch_operations'
    union all
    select 'int_events', sum(rows) from manifest
    where extract in ('card_transactions', 'online_operations', 'branch_operations')
),

actual as (
    select 'stg_clients' as model, count(*) as rows from {{ ref('stg_clients') }}
    union all
    select 'client_360', count(*) from {{ ref('client_360') }}
    union all
    select 'stg_card_transactions', count(*) from {{ ref('stg_card_transactions') }}
    union all
    select 'stg_online_operations', count(*) from {{ ref('stg_online_operations') }}
    union all
    select 'stg_branch_operations', count(*) from {{ ref('stg_branch_operations') }}
    union all
    select 'int_events', count(*) from {{ ref('int_events') }}
)

select expected.model, expected.rows as expected_rows, actual.rows as actual_rows
from expected
full outer join actual using (model)
where expected.rows is distinct from actual.rows
