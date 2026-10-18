# Strata

**Strata** is a behaviour-analytics pipeline for multi-channel retail-banking event data: typed
contracts at the door, a tested warehouse, an evidence ledger in which no published number is typed
by hand, and a robustness checker that re-stratifies every headline finding before anyone believes
it.

The pipeline is the product. Its reference instance is an anonymised, rolling-year extract of card,
online and branch activity of a sample of clients, provided by CIH Bank for an academic challenge.
A hand-specified synthetic generator is a first-class source, so the whole system runs, and is
tested, without the extract. The findings further down illustrate what the pipeline produces on its
reference instance; they are a by-product, not the point.

Strata replaces a hackathon script from 2024 whose bugs are documented openly in
[`legacy/README.md`](legacy/README.md).

> **Status: under construction.** The build follows [`ARCHITECTURE.md`](ARCHITECTURE.md) and
> [`MILESTONES.md`](MILESTONES.md); progress is logged in [`PROGRESS.md`](PROGRESS.md).

## What the pipeline does

| Stage | What happens | Where |
|---|---|---|
| Contracts and ingest | Every file is parsed explicitly and checked against a typed contract; a manifest fingerprints the bytes; the warehouse is reconciled with the files. | `strata/contracts/`, `strata/ingest/` |
| Data quality | Exact repeats, zero and sub-unit amounts, vocabularies, plausibility and join coverage are measured and reported as counts, never values. | `strata dq` |
| Warehouse | dbt on DuckDB: staging over code-map seeds, one event table for every channel, a dense client-month panel, monthly channel states, and marts for clients, silence features with time cuts, segments, monitoring and daily calendar series. | `dbt/` |
| Evidence ledger | Registered claims with client-bootstrap intervals, sample sizes, caveats and a data fingerprint. Documents are templates over the ledger, and CI fails on drift or on a number typed by hand. | `strata/ledger/` |
| Robustness checker | Each headline is recomputed inside strata of activity, age and tenure and labelled *holds*, *weakens* or *reverses*. | [method](docs/methods/robustness.md) |
| Reports | A private report of every claim, kept outside the repository, and a public demo on synthetic data. | `strata report` |
| Next | Calendar models (frequentist and Bayesian), a channel Markov chain with a migration simulator, time-validated risk models, an anomaly triage layer, an API, an orchestrated DAG and a dashboard. | [milestones](MILESTONES.md) |

## How it is checked

- `just check` runs formatting and lint (`ruff`), strict typing (`mypy`), the test suite (`pytest`
  with `hypothesis`, coverage gate) on synthetic data, the privacy scan and the ledger check. CI
  runs the same steps and builds the whole warehouse on synthetic data.
- Every dbt model and seed carries tests, including reconciliations against the ingest manifest and
  against the data-quality report.
- The robustness verdict rule was fixed before it met any data, and is tested on cases with a known
  answer (a textbook reversal, a uniform effect, an effect that vanishes inside strata).

## How a source plugs in

Today the pipeline reads two sources: the reference extract, through typed contracts, and the
synthetic generator, which writes files in exactly the same format. The next step (milestone M4.5)
makes the pipeline source-agnostic: a mapping file per source (`mappings/<source>.yaml`) loads any
extract with clients and dated channel events into canonical raw tables, dbt staging reads only
those tables, and a property-based test maps a generated, deliberately different schema onto the
same marts.

<!-- ledger:start -->
## What the reference instance shows

What the pipeline reports on its reference instance, rendered from the evidence ledger's
allow-listed claims (never typed by hand); see [the findings](docs/findings/index.md) for the
tables, methods, caveats and robustness checks behind each sentence.

- **Channels.** 51% of clients use both the branch and a
  digital channel, 29% only digital channels and
  15% only the branch
  (n = 1,000 clients; 2023-10-02 to 2024-10-02; real extract, allow-listed aggregate; data fingerprint 85887a44d747).
- **Age.** The branch-only share is
  6.9% in the <25 band and
  33% in the 65+ band, while the
  gap between the two gender codes is -1.5 percentage points.
  Robustness verdict: **holds**
  (n = 1,000 clients; 2023-10-02 to 2024-10-02; real extract, allow-listed aggregate; data fingerprint 85887a44d747).
- **The week.** On cards, the busiest day of the week carries
  1.10× the events of the quietest
  (n = 947 clients, 54,852 events; 2023-10-02 to 2024-10-02; real extract, allow-listed aggregate; data fingerprint 85887a44d747).
- **Prices.** Card purchase amounts show acceptable conformity with Benford's
  law; the share of leading nines differs from the law's expectation by
  +3.8 percentage points,
  the usual mark of psychological pricing (n = 515 clients, 14,496 events; 2023-10-02 to 2024-10-02; real extract, allow-listed aggregate; data fingerprint 85887a44d747).
<!-- ledger:end -->

## Privacy

The extract is reachable only through `STRATA_DATA_DIR` (set in a gitignored `.env.local` on the
machine that holds it); everything derived from it goes to `STRATA_PRIVATE_DIR`, outside the
repository; and a pre-commit guard blocks anything shaped like the extract. Only allow-listed
aggregates computed on enough clients are ever published. See [`PRIVACY.md`](PRIVACY.md).

## Quick start (synthetic data, no access to the extract needed)

```bash
uv sync                 # Python 3.12 environment from uv.lock
uv run pre-commit install
just check              # lint, types, tests with coverage, privacy scan, ledger check
just ledger             # synthetic extract -> warehouse -> claims -> demo report
uv run strata --help
```

[`uv`](https://docs.astral.sh/uv/) and [`just`](https://just.systems/) are the only tools to install
by hand (`uv tool install rust-just` provides `just`).

## Repository layout

| Path | Contents |
|---|---|
| `strata/` | The Python package: configuration, CLI, contracts, ingest, synthetic data, ledger, analysis, reports, privacy guard. |
| `dbt/` | The warehouse: seeds (code maps, bands, cutoffs, calendar), staging, intermediate models, marts and tests. |
| `docs/` | Architecture decision records, methods, and findings rendered from the ledger. |
| `tests/` | `pytest` and `hypothesis` suites; every test runs on synthetic data. |
| `legacy/` | The 2024 script and the 2025 package, frozen, with a list of their bugs. |

## Licence and data

The code is released under the MIT licence ([`LICENSE`](LICENSE)). The bank extract is
confidential and is not, and will never be, part of this repository.
