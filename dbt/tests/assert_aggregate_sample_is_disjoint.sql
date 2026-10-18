{{ config(severity='warn') }}
-- The aggregate file is a different sample: none of its identifiers should appear in the client
-- file. The 2024 analysis joined the two and silently got empty columns. A warning, not an error,
-- because a future extract could legitimately overlap; the warehouse never joins them either way.
select sample.client_id
from {{ ref('stg_aggregate_clients') }} as sample
inner join {{ ref('stg_clients') }} as clients using (client_id)
