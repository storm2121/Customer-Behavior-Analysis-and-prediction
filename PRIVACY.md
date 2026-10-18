# Privacy

Strata is developed on a confidential, anonymised extract of bank clients. This page explains, for
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

- **public** (the default) reads data produced by `strata synth`, which carries a `SYNTHETIC.json`
  marker, and writes under `.strata/` (gitignored). A public run refuses any data folder without the
  marker, so forgetting a flag fails closed. The marker records the hash of every
  file, so real files copied into a synthetic folder are refused as well.
- **private** (`strata --private …`) reads the real extract from `STRATA_DATA_DIR` and writes only
  under `STRATA_PRIVATE_DIR`, which defaults to a `private_out` folder next to the extract. Both
  variables are refused if they resolve inside the repository.

The location of the extract is known only through the environment. On the machine that holds
the extract, `STRATA_DATA_DIR` and `STRATA_PRIVATE_DIR` live in `.env.local` at the repository
root: gitignored, read by the settings so private runs need no manual export, and overridden by
real environment variables. No tracked file names the private locations, and the CLI prints them
as `<STRATA_PRIVATE_DIR>/…`. One exception is
cosmetic: dbt's own log lines name the folders it writes to, so a private `strata dbt` run shows the
private folder's path in the terminal (never in a file inside the repository). dbt's anonymous usage
statistics are switched off in the project and in the environment of every run.

## The pre-commit guard

`strata privacy-check` runs on every commit (installed by `just setup`) and in CI. It blocks:

- files named like an extract, and text files whose first line looks like an extract header;
- CSV files outside `tests/fixtures/` and `dbt/seeds/`, and any CSV with more than 200 rows;
- binary data formats (DuckDB, Parquet, pickles, spreadsheets) and private claim files;
- tokens shaped like a client identifier. On the machine that holds the extract the shape is learnt
  from the extract at run time and exact matches are reported as such; elsewhere a conservative
  pattern is used;
- JSON under `docs/ledger/` that contains a claim outside the allowlist, or a value computed on
  fewer than the minimum number of clients for publication;
- the literal private locations, taken from the environment and from `.env.local`;
- local configuration files (`.env`, `.env.local`): they stay untracked, and a forced add is
  refused;
- notebooks with outputs.

A finding names the file, the line and the rule. It never repeats the offending content.
Bypassing the hook (`git commit --no-verify`) is forbidden by the project rules.

## What may be published

Only claims listed in `strata/ledger/allowlist.py` are written to `docs/ledger/claims.public.json`,
by `strata --private ledger build --public`, and only if they pass the publication rules defined
once in that module and enforced both by the build and by the commit guard:

- every value is an aggregate over at least the publication floor of clients;
- a robustness result compares two groups that each reach the floor;
- a stratum of a robustness result keeps its numbers only when each of its two groups reaches the
  floor; otherwise only its label and whether it was used remain.

Two publication modes exist. A `full` claim is published with its aggregate values, intervals and
robustness result under the rules above; a value computed on fewer clients than the floor (the top
one per cent of spenders, say) is withheld with its table row. A `verdict_only` headline is
published by what the robustness checker concluded: its verdict, the names of the stratifiers, the
stratifiers within which the effect reverses, and an explanation in words. It carries no value, no
interval and no group size, and the build, the ledger check and the commit guard all refuse a
verdict-only record with any other field or with a digit in its words. Today `silence_by_channel`
is the only verdict-only headline; it is there to show the checker at work.

`--public` refuses to run on synthetic data, so the public file always holds real aggregates and
the synthetic demo can never overwrite it. Every other finding stays in the private report
(`strata --private report --variant private`, written under `STRATA_PRIVATE_DIR`).

The analysis modules follow the same zones. `strata --private analyse` writes its figures, its
claims and the simulator's inputs (transition counts between channel states, state counts and
branch-operation rates, all aggregates) under `STRATA_PRIVATE_DIR` and refuses any other location;
the figures are drawn from real data and are not published. Its terminal table of the sample
policy's bands is for the person running it, not a publication channel.

## Logs and errors

Log records go to standard error through `structlog`; fields named like identifiers are replaced
and identifier-shaped tokens inside messages are masked. The CLI never prints a traceback with local
variables. On a private run, the full traceback of a failure is written to the private folder and
the terminal only shows the exception type and a masked message.

## Tests

Every test in `tests/` runs on synthetic data generated during the session, in an environment
scrubbed of `STRATA_*` variables. Tests marked `private` exercise the real extract locally, are
skipped unless `--run-private` is passed, and never assert on values.

## If something slips through

Nothing leaves the machine without the owner's decision: the project rules forbid pushing. If a
private value is ever committed locally, rewrite the local history to remove it before anything is
shared, and record what happened (without the value) in `PROGRESS.md`.
