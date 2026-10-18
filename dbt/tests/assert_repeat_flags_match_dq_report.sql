-- One definition of an exact repeat, two implementations: staging flags the rows, `kashf dq`
-- counts them before dbt runs. Any disagreement between the two fails the build.
with flagged as (
    select 'card_transactions' as extract, count(*) filter (where is_repeat) as repeats
    from {{ ref('stg_card_transactions') }}
    union all
    select 'online_operations', count(*) filter (where is_repeat)
    from {{ ref('stg_online_operations') }}
    union all
    select 'branch_operations', count(*) filter (where is_repeat)
    from {{ ref('stg_branch_operations') }}
),

reported as (
    select extract, "count" as repeats
    from {{ source('raw', 'dq_report') }}
    where check_id = 'exact_repeats'
)

select flagged.extract, flagged.repeats as flagged, reported.repeats as reported
from flagged
full outer join reported using (extract)
where flagged.repeats is distinct from reported.repeats
