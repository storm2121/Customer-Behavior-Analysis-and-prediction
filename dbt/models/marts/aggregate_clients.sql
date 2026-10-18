-- The second sample, ready for the adoption models (M5): the cleaned columns, the shared age bands
-- and the two targets (online access, and active online use over the year).
with age_bands as (
    select * from {{ ref('bands_age') }}
)

select
    sample.*,
    age_bands.band as age_band,
    sample.has_online as target_has_online,
    coalesce(sample.ops_total >= {{ var('active_online_min_operations') }}, false)
        as target_active_online
from {{ ref('stg_aggregate_clients') }} as sample
left join age_bands
    on sample.age >= age_bands.lower and sample.age < age_bands.upper
