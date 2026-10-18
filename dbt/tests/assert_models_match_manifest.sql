-- Row counts of the warehouse against the ingest manifest: the canonical event table keeps every
-- row of the event files, each staging model keeps every canonical event of its channel, the event
-- table is their union, and the client marts hold every client exactly once. Written against the
-- manifest's roles, so it holds for every mapped source.
with manifest as (
    select role, rows from {{ source('raw', 'manifest') }}
),

events as (
    select channel, count(*) as rows from {{ source('raw', 'events') }} group by channel
),

expected as (
    select 'stg_clients' as model, sum(rows) as rows from manifest where role = 'clients'
    union all
    select 'client_360', sum(rows) from manifest where role = 'clients'
    union all
    select 'raw_events', sum(rows) from manifest where role = 'events'
    union all
    select 'stg_card_transactions', coalesce(sum(rows), 0) from events where channel = 'card'
    union all
    select 'stg_online_operations', coalesce(sum(rows), 0) from events where channel = 'online'
    union all
    select 'stg_branch_operations', coalesce(sum(rows), 0) from events where channel = 'branch'
    union all
    select 'int_events', sum(rows) from manifest where role = 'events'
),

actual as (
    select 'stg_clients' as model, count(*) as rows from {{ ref('stg_clients') }}
    union all
    select 'client_360', count(*) from {{ ref('client_360') }}
    union all
    select 'raw_events', count(*) from {{ source('raw', 'events') }}
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
