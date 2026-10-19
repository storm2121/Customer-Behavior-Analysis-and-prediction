# Data card: the reference instance

Status: complete. It describes the extract Strata was built around, what the pipeline assumes about
it, and what it may and may not be used for. The table of assumed and undocumented codes at the end
is generated from the seeds, and a test fails if it falls out of date. No figure in this card is
typed by hand: the sample size, the observation window and the data fingerprint travel with every
public claim in [`docs/ledger/claims.public.json`](ledger/claims.public.json) and the documents
rendered from it.

## Provenance

The reference instance is an anonymised extract provided by CIH Bank for an academic data
challenge. It is confidential: it is not redistributed, it is not part of this repository, and it
reaches the pipeline only through `STRATA_DATA_DIR` on the machine that holds it (see
[`PRIVACY.md`](../PRIVACY.md)). Nothing in the repository is read from it except the allow-listed
aggregates of the public ledger.

## Composition

Five files, read through typed contracts and the reference mapping (`mappings/`):

- **clients**: one row per client, with an opaque identifier, age, tenure, gender and marital codes,
  number of children and professional status;
- **card events**, **online-banking events** and **branch operations**: one row per event, with the
  client, a date, an amount where the channel has one, and the channel's own descriptors (merchant
  category and origin for cards, service for online banking, operation label and direction for the
  branch);
- **the aggregate file**: one row per client of a different sample, with yearly totals and flags of
  online-banking use, used only by the adoption models.

What the extract does not contain, and the pipeline therefore never assumes:

- no transaction identifier and no time of day, so an exact repeat of a row is either a double
  posting or a genuine second operation, and is flagged rather than dropped;
- no documentation of the code values, so several meanings below are inferred;
- no closing date of accounts, so silence (a quarter without any event) is not attrition;
- no balances, products or incomes, so nothing here measures wealth or credit risk.

## Collection and coverage

A sample of the bank's clients, with their activity over a single rolling year. How the sample was
drawn is not documented. The aggregate file describes clients outside that sample, so it can never
be joined to the event files (a dbt test asserts that the two samples are disjoint). Amounts are in
dirhams. Calendar effects rest on one year: each month and holiday is seen once, so seasonality
cannot be told apart from trend, and every calendar result says so.

## Known biases and limitations

- **Selection.** An unknown sampling design: shares describe the sample, not the bank's clientele.
- **Channels.** The branch file holds the operations that reach an account; cash handled without an
  account operation is invisible. Card events mix purchases, withdrawals and service operations,
  which the code maps separate.
- **Codes.** Gender and marital codes, the aggregate flags and some branch labels have meanings
  inferred from their wording or from usual practice (table below); claims that depend on one say
  so in their caveats.
- **Quality.** Exact repeats, zero and sub-unit amounts are measured by `strata dq` and reported as
  counts, never values; the warehouse keeps them flagged so that every analysis decides explicitly.
- **Size.** The sample is small enough that per-client and small-group numbers would identify
  people; publication is therefore limited to allow-listed aggregates over enough clients, and the
  private twin pays for its privacy with fidelity ([the twin's decision record](adr/0013-private-twin.md)).

## Intended uses

- The reference instance of a public, source-agnostic pipeline: contracts, warehouse, ledger,
  models, monitoring, serving and dashboard are built and tested against its shapes.
- Aggregate findings, published only through the allow-list, each with its interval, sample size,
  caveats and robustness verdict.
- Private analysis on the machine that holds the extract, with outputs under `STRATA_PRIVATE_DIR`.

## Uses it must not serve

- Any decision about an individual client: scores and alerts rank cases for a person to review and
  are never a verdict on anyone.
- Credit, pricing or eligibility decisions, for which the extract holds neither the data nor the
  consent.
- Re-identification, or publishing any per-client or small-group output.
- Treating results of the synthetic demo as evidence about the bank: they show that the pipeline
  runs, nothing more.
- Using the private twin outside the machine that holds the extract without a review of its
  metrics and a decision record of its own.

## Maintenance

The assumptions below stand until the challenge's data dictionary confirms or corrects them;
Mohamed may hold it. Correcting a meaning is a change to its seed, followed by
`python -m strata.contracts.data_card`, a rebuild and `just publish-claims` on the machine that
holds the extract.

## Assumptions and undocumented codes

Every meaning below is used by the pipeline but was not documented by the bank, or is a calendar
date taken from an official calendar that still has to be confirmed. Each one is tagged in its seed
(`dbt/seeds/`), so the table is generated from the seeds by `python -m strata.contracts.data_card`
and a test fails if it falls out of date.

- **assumed**: a meaning inferred from the label's wording, from the usual coding of such a field,
  or from an official calendar; used as stated until confirmed.
- **undocumented**: no meaning could be inferred; the code is kept as a category and never
  interpreted.

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
