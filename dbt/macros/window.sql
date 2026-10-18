{#- The rolling year of the extract, from the project variables (one definition for every model). -#}
{% macro window_start() -%}
    date '{{ var("window_start") }}'
{%- endmacro %}

{% macro window_end() -%}
    date '{{ var("window_end") }}'
{%- endmacro %}

{% macro card_end() -%}
    date '{{ var("card_end") }}'
{%- endmacro %}

{#- Channel profile of a set of events, from per-channel counts. -#}
{% macro channel_profile(n_card, n_online, n_branch) -%}
    case
        when coalesce({{ n_card }}, 0) + coalesce({{ n_online }}, 0) + coalesce({{ n_branch }}, 0) = 0 then 'none'
        when coalesce({{ n_branch }}, 0) > 0 and coalesce({{ n_card }}, 0) + coalesce({{ n_online }}, 0) = 0 then 'branch_only'
        when coalesce({{ n_branch }}, 0) = 0 then 'digital_only'
        else 'mixed'
    end
{%- endmacro %}
