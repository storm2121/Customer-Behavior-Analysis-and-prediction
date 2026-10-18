-- The channel state of every client in every month (none, branch_only, digital_only, mixed) and
-- the state of the month before, which is what the Markov chain counts.
with states as (
    select
        client_id,
        month_start,
        is_partial_month,
        n_events,
        {{ channel_profile('n_card', 'n_online', 'n_branch') }} as channel_state
    from {{ ref('int_client_month') }}
)

select
    *,
    lag(channel_state) over (partition by client_id order by month_start) as previous_state
from states
