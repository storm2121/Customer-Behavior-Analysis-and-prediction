# ADR 0008: private runs write nothing inside the working tree

Status: accepted (executing agent, M0, 2026-10-18). Deviates from ARCHITECTURE.md sections 5.2 and 7.
ADR 0007, numbered before this one, records the Bayesian backend choice (ARCHITECTURE.md section 9).

## Context

ARCHITECTURE.md places the warehouse at `STRATA_DB` (default `.strata/strata.duckdb`) and the private
claims at `docs/ledger/claims.private.json`, both gitignored. When the warehouse is built from the
real extract, both locations would hold data derived from real rows inside the repository's working
tree. AGENT_RULES.md section 1.3 forbids exactly that: everything derived from real rows, except the
allow-listed public claims, goes under `STRATA_PRIVATE_DIR`. A gitignore entry is a second line of
defence, not the boundary (ADR 0004). The same applies to dbt's `target/` and `logs/` folders, which
record compiled SQL, row counts and test results of a private run.

## Decision

Every command runs in one of two zones:

- **public** (default): reads only data folders that carry the `SYNTHETIC.json` marker written by
  `strata synth`; writes the warehouse to `STRATA_DB` and everything else under `.strata/`.
- **private** (`strata --private …` or `STRATA_ZONE=private`): reads the extract from
  `STRATA_DATA_DIR` and writes the warehouse, dbt target and logs, ledger, reports, figures and
  error logs under `STRATA_PRIVATE_DIR`. The private claims file is
  `<STRATA_PRIVATE_DIR>/ledger/claims.private.json`.

The zone is never inferred from the presence of data. A public run refuses a folder without the
synthetic marker, so a forgotten `--private` fails closed instead of writing real rows into
`.strata/`. The only private-zone output that enters the repository is
`docs/ledger/claims.public.json`, written through the allowlist by `strata ledger build --public`.

## Consequences

- Private commands take an explicit flag; the documented commands in MILESTONES.md read
  `strata --private ingest` rather than `strata ingest --data-dir "$STRATA_DATA_DIR"`.
- Public and private dbt artefacts never mix, so a public `dbt docs generate` cannot pick up the
  catalogue of a private run.
- `claims.private.json` stays in `.gitignore` and on the guard's block list as a second line.
