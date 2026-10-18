# Data card (draft)

Status: a draft started in milestone M4 for one purpose, to keep every assumption about the
reference extract visible in one place. The full card (provenance, collection, coverage, known
biases, intended and unintended uses) is completed in milestone M10.

## The reference instance

The reference instance is the anonymised CIH Bank extract of the data challenge: one file of
clients, three files of dated channel events (card, online banking, branch) and one aggregate file
describing a different sample of clients. It reaches the pipeline only through `STRATA_DATA_DIR`
on the machine that holds it (see [`PRIVACY.md`](../PRIVACY.md)); nothing in this repository is
read from it except the allow-listed aggregates of `docs/ledger/claims.public.json`.

What the extract does not contain, and the pipeline therefore never assumes:

- no transaction identifier and no time of day, so an exact repeat of a row is either a double
  posting or a genuine second operation, and is flagged rather than dropped;
- no documentation of the code values, so several meanings below are inferred;
- no closing date of accounts, so silence (a quarter without any event) is not attrition.

## Assumptions and undocumented codes

Every meaning below is used by the pipeline but was not documented by the bank, or is a calendar
date taken from an official calendar that still has to be confirmed. Each one is tagged in its seed
(`dbt/seeds/`), so the table is generated from the seeds by `python -m strata.contracts.data_card`
and a test fails if it falls out of date.

- **assumed**: a meaning inferred from the label's wording, from the usual coding of such a field,
  or from an official calendar; used as stated until confirmed.
- **undocumented**: no meaning could be inferred; the code is kept as a category and never
  interpreted.

Mohamed may hold the challenge's data dictionary; until it is checked, these assumptions stand, and
every claim that depends on one of them says so in its caveats.

<!-- assumptions:start (generated from dbt/seeds; do not edit by hand) -->
| Seed | Field | Code | Meaning used | Status | Evidence |
|---|---|---|---|---|---|
| `codes_branch_operation` | branch operation label | `RESITUTION DU SOLDE CARTE E-SHOPPING` | eshopping_card_refund: category card, direction C | assumed | rare label; meaning read from its wording |
| `codes_branch_operation` | branch operation label | `RECHARGE CARTE EN ESPECE` | card_topup_cash: category card, direction C | assumed | rare label; meaning read from its wording |
| `codes_branch_operation` | branch operation label | `ACHAT BILLET MONO-DEVISE PAR UN CLIENT PAR CREDIT DE COMPTE` | fx_notes_account: category fx, direction C | assumed | rare label; meaning read from its wording |
| `codes_branch_operation` | branch operation label | `ACHAT BILLET MONO-DEVISE PAR UN CLIENT EN ESPECES` | fx_notes_cash: category fx, direction D | assumed | rare label; meaning read from its wording |
| `codes_branch_operation` | branch operation label | `ANNULATION VERSEMENT SUR COMPTE` | deposit_reversal: category adjustment, direction D | assumed | rare label; meaning read from its wording |
| `codes_branch_operation` | branch operation label | `ANNULATION RETRAIT CHEQUE` | cheque_withdrawal_reversal: category adjustment, direction C | assumed | rare label; meaning read from its wording |
| `codes_branch_operation` | branch operation label | `INITIATION REGULARISATION INCIDENT DE PAIEMENT` | payment_incident_regularisation: category adjustment, direction D | assumed | rare label; meaning read from its wording |
| `codes_branch_operation` | branch operation label | `BLOCAGE DE PROVISION` | funds_hold: category adjustment, direction D | assumed | rare label; meaning read from its wording |
| `codes_branch_operation` | branch operation label | `REGULARISATION RETRACT DEPOT DE CASH` | cash_deposit_regularisation: category adjustment, direction C | assumed | rare label; meaning read from its wording |
| `codes_client` | genre | `1` | male | assumed | usual coding of the field; not documented by the bank |
| `codes_client` | genre | `2` | female | assumed | usual coding of the field; not documented by the bank |
| `codes_client` | situation_familiale | `1` | single | assumed | usual coding of the field; not documented by the bank |
| `codes_client` | situation_familiale | `2` | married | assumed | usual coding of the field; not documented by the bank |
| `codes_client` | situation_familiale | `3` | divorced | assumed | usual coding of the field; not documented by the bank |
| `codes_client` | situation_familiale | `4` | widowed | assumed | usual coding of the field; not documented by the bank |
| `codes_aggregate_flags` | flag_contrat_actif | `1` | yes | assumed | aggregate file; not documented by the bank |
| `codes_aggregate_flags` | flag_contrat_actif | `2` | no | assumed | aggregate file; not documented by the bank |
| `codes_aggregate_flags` | transaction_international | `0` | no | assumed | aggregate file; not documented by the bank |
| `codes_aggregate_flags` | transaction_international | `1` | yes | assumed | aggregate file; not documented by the bank |
| `codes_aggregate_flags` | civilite | `0` | no documented meaning; kept as a category | undocumented | aggregate file; not documented by the bank |
| `codes_aggregate_flags` | civilite | `1` | no documented meaning; kept as a category | undocumented | aggregate file; not documented by the bank |
| `codes_aggregate_flags` | flag_debit_credit | `-1` | no documented meaning; kept as a category | undocumented | aggregate file; not documented by the bank |
| `codes_aggregate_flags` | flag_debit_credit | `0` | no documented meaning; kept as a category | undocumented | aggregate file; not documented by the bank |
| `codes_aggregate_flags` | flag_debit_credit | `1` | no documented meaning; kept as a category | undocumented | aggregate file; not documented by the bank |
| `codes_aggregate_flags` | flag_debit_credit | `2` | no documented meaning; kept as a category | undocumented | aggregate file; not documented by the bank |
| `calendar_periods` | calendar period | `ramadan` | Ramadan 1445, 2024-03-12 to 2024-04-09 | assumed | Moroccan moon sighting (architect; Mohamed to confirm) |
| `calendar_periods` | calendar period | `eid_al_fitr` | Eid al-Fitr, 2024-04-10 to 2024-04-11 | assumed | Moroccan moon sighting (architect; Mohamed to confirm) |
| `calendar_periods` | calendar period | `eid_al_adha` | Eid al-Adha, 2024-06-17 to 2024-06-18 | assumed | Moroccan moon sighting (architect; Mohamed to confirm) |
| `calendar_periods` | calendar period | `pre_eid_al_adha_week` | Week before Eid al-Adha, 2024-06-10 to 2024-06-16 | assumed | the seven days before Eid al-Adha |
| `bank_holidays` | bank holiday (national) | `2023-11-06` | Green March | assumed | fixed-date national holiday (architect; Mohamed to confirm) |
| `bank_holidays` | bank holiday (national) | `2023-11-18` | Independence Day | assumed | fixed-date national holiday (architect; Mohamed to confirm) |
| `bank_holidays` | bank holiday (national) | `2024-01-01` | New Year | assumed | fixed-date national holiday (architect; Mohamed to confirm) |
| `bank_holidays` | bank holiday (national) | `2024-01-11` | Proclamation of Independence | assumed | fixed-date national holiday (architect; Mohamed to confirm) |
| `bank_holidays` | bank holiday (national) | `2024-01-14` | Amazigh New Year | assumed | official since 2024 by royal decision; not in the architect list; Mohamed to confirm |
| `bank_holidays` | bank holiday (national) | `2024-05-01` | Labour Day | assumed | fixed-date national holiday (architect; Mohamed to confirm) |
| `bank_holidays` | bank holiday (religious) | `2024-07-08` | Islamic New Year 1446 | assumed | Moroccan moon sighting (architect; Mohamed to confirm) |
| `bank_holidays` | bank holiday (national) | `2024-07-30` | Throne Day | assumed | fixed-date national holiday (architect; Mohamed to confirm) |
| `bank_holidays` | bank holiday (national) | `2024-08-14` | Oued Ed-Dahab Day | assumed | fixed-date national holiday (architect; Mohamed to confirm) |
| `bank_holidays` | bank holiday (national) | `2024-08-20` | Revolution of the King and the People | assumed | fixed-date national holiday (architect; Mohamed to confirm) |
| `bank_holidays` | bank holiday (national) | `2024-08-21` | Youth Day | assumed | fixed-date national holiday (architect; Mohamed to confirm) |
| `bank_holidays` | bank holiday (religious) | `2024-09-15` | Mawlid | assumed | Moroccan moon sighting (architect; Mohamed to confirm) |
| `bank_holidays` | bank holiday (religious) | `2024-09-16` | Mawlid (second day) | assumed | Moroccan moon sighting (architect; Mohamed to confirm) |
<!-- assumptions:end -->
