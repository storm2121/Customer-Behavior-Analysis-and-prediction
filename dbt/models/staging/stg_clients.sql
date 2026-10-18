-- Clients from the canonical table, with the assumed code meanings and the shared bands.
with source as (
    select * from {{ source('raw', 'clients') }}
),

codes as (
    select column_name, code, label from {{ ref('codes_client') }}
),

age_bands as (
    select * from {{ ref('bands_age') }}
),

tenure_bands as (
    select * from {{ ref('bands_tenure') }}
)

select
    source.client_id,
    source.tenure_years,
    source.age,
    source.gender_code,
    gender.label as gender,
    source.marital_code,
    marital.label as marital_status,
    source.children,
    source.sector_code,
    source.profession_code,
    source.socio_category_code,
    (
        source.sector_code is null
        and source.profession_code is null
        and source.socio_category_code is null
    ) as profession_missing,
    age_bands.band as age_band,
    tenure_bands.band as tenure_band
from source
left join codes as gender
    on gender.column_name = 'genre' and gender.code = source.gender_code
left join codes as marital
    on marital.column_name = 'situation_familiale' and marital.code = source.marital_code
left join age_bands
    on source.age >= age_bands.lower and source.age < age_bands.upper
left join tenure_bands
    on source.tenure_years >= tenure_bands.lower
    and source.tenure_years < tenure_bands.upper
