"""Static description of the five extracts: names, file format and column order.

This module is the single source of truth for the shape of the source files. The privacy guard uses
it to recognise an extract header, the public generator uses it to write files that look exactly
like the originals, and the ingest contracts use it to validate them. It holds schema only, never a
value from the data.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from datetime import date

ENCODING = "latin-1"
"""The extracts are Latin-1; two column names and several labels carry accented characters."""

LINE_TERMINATOR = "\r\n"
DATE_FORMAT = "%Y-%m-%dT%H:%M"
"""Dates are written ``YYYY-MM-DDTHH:MM`` and the time is always midnight: no time of day exists."""

WINDOW_START = date(2023, 10, 2)
WINDOW_END = date(2024, 10, 2)
"""The rolling year covered by the event files (inclusive bounds)."""


class Extract(enum.StrEnum):
    """Logical names used everywhere in code instead of the original file names."""

    CLIENTS = "clients"
    CARD = "card_transactions"
    ONLINE = "online_operations"
    BRANCH = "branch_operations"
    AGGREGATE = "aggregate_clients"


@dataclass(frozen=True, slots=True)
class ExtractSpec:
    """Shape of one source file.

    ``date_column`` is ``None`` for the two client-grain files. ``event`` is true for the three
    files with one row per event, which are the ones the warehouse unions into ``int_events``.
    """

    name: Extract
    filename: str
    columns: tuple[str, ...]
    id_column: str
    date_column: str | None
    grain: str

    @property
    def event(self) -> bool:
        return self.date_column is not None

    @property
    def raw_table(self) -> str:
        return f"raw.{self.name.value}"


CLIENTS = ExtractSpec(
    name=Extract.CLIENTS,
    filename="echantillons_clients(in).csv",
    columns=(
        "identifiant_client",
        "anciennete_client",
        "age",
        "genre",
        "situation_familiale",
        "nombre_d_enfants",
        "secteur_professionnel",
        "profession",
        "categorie_socio_professionelle",
    ),
    id_column="identifiant_client",
    date_column=None,
    grain="one row per client",
)

CARD = ExtractSpec(
    name=Extract.CARD,
    filename="transactions_monetiques_annee_glissante_anonymise(in).csv",
    columns=(
        "identifiant_client",
        "date_transaction",
        "montant_local",
        "libelle_nature_transaction",
        "type_origine",
    ),
    id_column="identifiant_client",
    date_column="date_transaction",
    grain="one row per card transaction",
)

ONLINE = ExtractSpec(
    name=Extract.ONLINE,
    filename="operations_en_ligne_annee_glissante_anonymise(in).csv",
    columns=("identifiant_client", "service", "operation_date_time", "montant_transaction"),
    id_column="identifiant_client",
    date_column="operation_date_time",
    grain="one row per online operation",
)

BRANCH = ExtractSpec(
    name=Extract.BRANCH,
    filename="operations_en_agence_annee_glissante_anonymise(in).csv",
    columns=(
        "identifiant_client",
        "date_operation",
        "montant_operation",
        "sens_operation",
        "operation",
        "devise",
    ),
    id_column="identifiant_client",
    date_column="date_operation",
    grain="one row per branch operation",
)

AGGREGATE = ExtractSpec(
    name=Extract.AGGREGATE,
    filename="al_akhawayn_dataset_v2(in).csv",
    columns=(
        "identifier",
        "duree_creation_client_en_annee",
        "age",
        "civilite",
        "pays_de_naissance",
        "code_pays_de_residence",
        "situation_familiale",
        "nombre_d_enfants",
        "situation_professionelle",
        "lieu_creation",
        "a_email",
        "resident_status",
        "flag_profession",
        "client_has_cih_online",
        "flag_contrat_actif",
        "virement_ponctuel",
        "paiement_facture_recharge_telecom_paiement_vignette",
        "virement_permanent",
        "operations_sur_carte",
        "transfert_cash_express",
        "recharge_eshopping",
        "debit_carte",
        "recharge_binatna",
        "image_cheque_emis",
        "nombre_operations",
        "moyenne_operation",
        "ecart_type_operation",
        "mediane_operation",
        "minimum_operation",
        "maximum_operation",
        "nombre_mois_premiere_operation",
        "nombre_mois_derniere_operation",
        "nombre_total_credits",
        "credit_immobilier",
        "credit_consommation",
        "flag_debit_credit",
        "carte_cih_utilisee_dans_un_gab_cih",
        "carte_cih_utilisee_chez_un_commerçant_national",
        "carte_cih_utilisee_chez_un_commerçant_international",
        "carte_nationale_hors_carte_cih_utilisee_dans_un_gab_cih",
        "carte_internationale_hors_carte_cih_utilisee_dans_un_gab_cih",
        "deposit",
        "withdrawal",
        "mastercardmoney_send",
        "refund",
        "bill_payment",
        "achat_sur_gab",
        "visa_money_transfert_load",
        "achats_e_commerce",
        "purchases_and_services",
        "paiement_vignette",
        "nombre_transaction",
        "moyenne_montant_transaction",
        "ecart_type_montant_transaction",
        "mediane_montant_transaction",
        "minimum_montant_transaction",
        "maximum_montant_transaction",
        "transaction_international",
        "nombre_mois_premiere_transaction",
        "nombre_mois_derniere_transaction",
    ),
    id_column="identifier",
    date_column=None,
    grain="one row per client of a different sample, pre-aggregated over the year",
)

EXTRACTS: tuple[ExtractSpec, ...] = (CLIENTS, CARD, ONLINE, BRANCH, AGGREGATE)
BY_NAME: dict[Extract, ExtractSpec] = {spec.name: spec for spec in EXTRACTS}
EVENT_EXTRACTS: tuple[ExtractSpec, ...] = tuple(spec for spec in EXTRACTS if spec.event)


def spec_for(name: Extract | str) -> ExtractSpec:
    """Look up a spec by its logical name."""
    return BY_NAME[Extract(name)]
