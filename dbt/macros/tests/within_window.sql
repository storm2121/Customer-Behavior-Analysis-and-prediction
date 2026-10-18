{% test within_window(model, column_name) %}
    select * from {{ model }}
    where {{ column_name }} < {{ window_start() }} or {{ column_name }} > {{ window_end() }}
{% endtest %}
