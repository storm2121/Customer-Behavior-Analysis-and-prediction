# Source mappings

A mapping tells `strata ingest` how a source becomes the canonical raw tables that everything else
reads (ARCHITECTURE section 5.0, ADR 0010):

```bash
uv run strata ingest --data-dir <folder> --mapping <name or path>
```

Two mappings ship. `reference_cih.yaml` describes the reference instance and reads its five files
through the typed contracts of milestone M1. `synthetic.yaml` describes the public generator's
output, which has exactly the same format. The private zone uses the first by default, the public
zone the second.

## The format

```yaml
source: my_bank                # a name; recorded in the manifest
description: >
  What the source is.
encoding: utf-8                # defaults for every file (each file may override them)
delimiter: ";"
decimal: ","
date_format: "%d/%m/%Y"

clients:                       # one row per client
  file: clients.csv
  columns:                     # canonical name: source column
    client_id: customer
    age: age_years

events:                        # one or more files of dated events
  - file: card.csv
    channel: card              # card, online or branch
    columns:
      client_id: customer
      ts: posted_on
      amount: value
      kind: category           # translated through kind_codes into the seeds' labels
    kind_codes:
      POS: "PURCHASES & SERVICES"
  - file: other_events.csv     # or one file holding several channels
    channel_column: medium
    channel_codes: {W: online, A: branch}
    columns: {client_id: customer, ts: when, amount: value, kind: what, direction: sign}
    direction_codes: {"+": C, "-": D}

client_aggregates:             # optional: a pre-aggregated sample of clients
  file: summary.csv
  columns:
    client_id: id
```

The canonical columns are defined in `strata/contracts/mapping.py`: `client_id` and the client
attributes (`CLIENT_TYPES`), the event columns (`EVENT_TYPES`: `ts`, `channel`, `kind`, `amount`,
`direction`, `origin`, `currency`) and the aggregate attributes (`AGGREGATE_TYPES`). A canonical
column the source does not have is left empty. Every mapped column must be present and must parse
to its canonical type, and every code must translate, or the ingest stops with the count of
offending values (never the values themselves).

`kind` holds the label the seeds use for the event's category (`dbt/seeds/codes_card_category.csv`,
`codes_online_service.csv`, `codes_branch_operation.csv`), so a new source inherits every meaning
the seeds define, including the ones tagged `assumed` (`docs/data_card.md`).
