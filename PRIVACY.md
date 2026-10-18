# Privacy

Kashf is developed on a confidential, anonymised extract of bank clients. This page explains, for
any reader of the repository, why none of that data is here and how the code makes sure it never
arrives. The decision record is [ADR 0004](docs/adr/0004-privacy-boundary.md), refined by
[ADR 0008](docs/adr/0008-private-zone-outside-the-working-tree.md).

## What is and is not in this repository

| In the repository | Never in the repository |
|---|---|
| Code, tests, dbt models, documentation | The five source files of the extract |
| A hand-specified synthetic generator and everything it produces | Any row, identifier or per-client value derived from the extract |
| `docs/ledger/claims.public.json`: aggregate claims whose ids are on the allowlist | Every other claim, the private report, models fitted on real data |
| Models trained on synthetic data | The warehouse built from the real extract, its logs and its dbt artefacts |

## Two zones

Every command runs in one of two zones, chosen explicitly:

- **public** (the default) reads data produced by `kashf synth`, which carries a `SYNTHETIC.json`
  marker, and writes under `.kashf/` (gitignored). A public run refuses any data folder without the
  marker, so forgetting a flag fails closed. The marker records the hash of every
  file, so real files copied into a synthetic folder are refused as well.
- **private** (`kashf --private …`) reads the real extract from `KASHF_DATA_DIR` and writes only
  under `KASHF_PRIVATE_DIR`, which defaults to a `private_out` folder next to the extract. Both
  variables are refused if they resolve inside the repository.

The location of the extract is known only through the environment. No file in the repository
names it, and the CLI prints private locations as `<KASHF_PRIVATE_DIR>/…`.

## The pre-commit guard

`kashf privacy-check` runs on every commit (installed by `just setup`) and in CI. It blocks:

- files named like an extract, and text files whose first line looks like an extract header;
- CSV files outside `tests/fixtures/` and `dbt/seeds/`, and any CSV with more than 200 rows;
- binary data formats (DuckDB, Parquet, pickles, spreadsheets) and private claim files;
- tokens shaped like a client identifier. On the machine that holds the extract the shape is learnt
  from the extract at run time and exact matches are reported as such; elsewhere a conservative
  pattern is used;
- JSON under `docs/ledger/` that contains a claim outside the allowlist, or a value computed on
  fewer than the minimum number of clients for publication;
- the literal private locations taken from the environment;
- notebooks with outputs.

A finding names the file, the line and the rule. It never repeats the offending content.
Bypassing the hook (`git commit --no-verify`) is forbidden by the project rules.

## What may be published

Only claims listed in `kashf/ledger/allowlist.py` are written to `docs/ledger/claims.public.json`,
and only if every value in them is an aggregate over at least the minimum number of clients set in
that module. Every other finding stays in the private report.

## Logs and errors

Log records go to standard error through `structlog`; fields named like identifiers are replaced
and identifier-shaped tokens inside messages are masked. The CLI never prints a traceback with local
variables. On a private run, the full traceback of a failure is written to the private folder and
the terminal only shows the exception type and a masked message.

## Tests

Every test in `tests/` runs on synthetic data generated during the session, in an environment
scrubbed of `KASHF_*` variables. Tests marked `private` exercise the real extract locally, are
skipped unless `--run-private` is passed, and never assert on values.

## If something slips through

Nothing leaves the machine without the owner's decision: the project rules forbid pushing. If a
private value is ever committed locally, rewrite the local history to remove it before anything is
shared, and record what happened (without the value) in `PROGRESS.md`.
