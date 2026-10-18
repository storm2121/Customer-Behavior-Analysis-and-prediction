{#- Use the configured schema as-is (staging, intermediate, marts, ref) instead of prefixing it with
    the target schema, so Python reads `marts.client_360` on every machine. -#}
{% macro generate_schema_name(custom_schema_name, node) -%}
    {%- if custom_schema_name is none -%}
        {{ target.schema }}
    {%- else -%}
        {{ custom_schema_name | trim }}
    {%- endif -%}
{%- endmacro %}
