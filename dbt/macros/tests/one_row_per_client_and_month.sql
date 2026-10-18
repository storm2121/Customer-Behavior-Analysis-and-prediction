{#- A client-month panel must hold every client in every month: its partition sums to the client
    count in each month (the "state partition" check of ARCHITECTURE section 5.2). -#}
{% test one_row_per_client_and_month(model, month_column) %}
    with expected as (select count(*) as n from {{ ref('stg_clients') }}),
    actual as (
        select {{ month_column }} as month_start, count(*) as n, count(distinct client_id) as clients
        from {{ model }}
        group by 1
    )
    select actual.*
    from actual cross join expected
    where actual.n != expected.n or actual.clients != expected.n
{% endtest %}
