-- Client file with English names, the assumed code meanings and the shared bands.
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
    source.identifiant_client as client_id,
    source.anciennete_client as tenure_years,
    source.age,
    source.genre as gender_code,
    gender.label as gender,
    source.situation_familiale as marital_code,
    marital.label as marital_status,
    source.nombre_d_enfants as children,
    source.secteur_professionnel as sector_code,
    source.profession as profession_code,
    source.categorie_socio_professionelle as socio_category_code,
    (
        source.secteur_professionnel is null
        and source.profession is null
        and source.categorie_socio_professionelle is null
    ) as profession_missing,
    age_bands.band as age_band,
    tenure_bands.band as tenure_band
from source
left join codes as gender
    on gender.column_name = 'genre' and gender.code = source.genre
left join codes as marital
    on marital.column_name = 'situation_familiale' and marital.code = source.situation_familiale
left join age_bands
    on source.age >= age_bands.lower and source.age < age_bands.upper
left join tenure_bands
    on source.anciennete_client >= tenure_bands.lower
    and source.anciennete_client < tenure_bands.upper
