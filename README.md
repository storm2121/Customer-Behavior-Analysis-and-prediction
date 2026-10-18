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
| Analysis modules | Calendar effects from a negative-binomial regression with client-bootstrap intervals and a PyMC twin with posterior predictive checks; STL seasonality and the December wave split into breadth and depth; Lorenz, Gini and RFM scores per channel; Benford tests per category; a Dirichlet-posterior Markov chain of channel states with a what-if simulator. Each module writes ledger claims and figures. | `strata analyse`, [methods](docs/methods/) |
| Risk models | Silence risk (baselines, a logistic GLM with odds ratios, LightGBM tuned on the earliest cutoff, isotonic and Platt calibration, SHAP), survival of the gaps between active days (Kaplan-Meier, log-rank, Cox with client-clustered errors), online adoption on the aggregate sample, and behavioural segments (Gaussian mixtures by BIC, bootstrap stability). Time-based backtests, a leakage test, client-bootstrap intervals on every metric, MLflow runs and generated model cards. | `strata train`, [model cards](docs/model_cards/) |
| Reports | A private report of every claim, kept outside the repository, and a public demo on synthetic data. | `strata report` |
| Monitoring | Triage rules configured in YAML (large cash deposits, structuring, velocity, pass-through, dormant-then-burst, cash intensity), an Isolation Forest, an empirical-Bayes model of each client's amounts and per-client Benford tests, combined into a ranked queue under an alert budget, with reasons on every alert; analyst feedback in a PostgreSQL or SQLite review store with Alembic migrations. Triage for people to review, never an accusation. | `strata anomaly`, [method](docs/methods/anomaly.md) |
| Serving | A FastAPI service: client 360, silence risk with reasons, segments, adoption deciles, the alert queue with feedback, calendar effects, the policy simulator and the public claims; Prometheus metrics by route; OpenAPI docs; Docker Compose with PostgreSQL and the MLflow UI. | `strata serve`, `docker compose up` |
| Next | An orchestrated DAG and a dashboard. | [milestones](MILESTONES.md) |

## How it is checked

- `just check` runs formatting and lint (`ruff`), strict typing (`mypy`), the test suite (`pytest`
  with `hypothesis`, coverage gate) on synthetic data, the privacy scan and the ledger check. CI
  runs the same steps and builds the whole warehouse on synthetic data.
- Every dbt model and seed carries tests, including reconciliations against the ingest manifest and
  against the data-quality report.
- The robustness verdict rule was fixed before it met any data, and is tested on cases with a known
  answer (a textbook reversal, a uniform effect, an effect that vanishes inside strata).

## How a source plugs in

Any extract with clients and dated channel events plugs in through a mapping file
(`mappings/<source>.yaml`, described in [`mappings/README.md`](mappings/README.md)): which file plays
which role, its encoding, delimiter, decimal mark and date format, which source column feeds which
canonical column, and how its codes translate into the vocabulary of the seeds. `strata ingest
--mapping <file>` loads it into three canonical tables, `raw.clients`, `raw.events` and
`raw.client_aggregates`, and dbt staging reads only those, so every mart, claim and model works
unchanged on a new source. Two mappings ship: the reference instance (read through its typed
contracts) and the synthetic generator. Two tests hold the layer to its promise: the reference
mapping reproduces the manifest, the data-quality report and every model's row count of the pipeline
as it was before mappings existed, and a property-based test maps a generated, deliberately
different schema onto the same marts ([ADR 0010](docs/adr/0010-canonical-layer.md)).

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
- **Concentration.** The Gini coefficient of card spend is
  0.76 (n = 646 clients; 2023-10-02 to 2024-10-02; real extract, allow-listed aggregate; data fingerprint 85887a44d747).
- **Robustness at work.** Are branch-only clients more likely than digital-only clients to have no event in the quarter after the observation window, and does that survive holding activity fixed?
  Before stratification the silence rate is higher for branch-only clients than for digital-only clients. Holding activity level (tercile of events) fixed, the difference reverses its sign. Holding age band fixed, the difference keeps its sign. Holding tenure band fixed, the difference keeps its sign. Verdict: reverses. Mechanism: branch-only clients have fewer events in the observation window than digital-only clients; with few events a quarter without any is likely by chance, so the crude gap mostly measures activity, not channel.
  (verdict only, no value published; 2023-10-02 to 2024-10-02; real extract; data fingerprint 85887a44d747; the numbers stay private).
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
just analyse            # analysis modules: claims, figures and the simulator's bands
just train              # risk models, the public registry and the model cards (synthetic)
just anomaly            # the monitoring layer's ranked review queue (synthetic)
uv run strata --help
```

[`uv`](https://docs.astral.sh/uv/) and [`just`](https://just.systems/) are the only tools to install
by hand (`uv tool install rust-just` provides `just`).

### The whole stack, on synthetic data

```bash
docker compose up --build
```

The API answers on `http://localhost:8000` (documentation at `/docs`, metrics at `/metrics`),
PostgreSQL on 5432, the MLflow UI on 5000 and the dashboard placeholder on 5173. The image builds
the synthetic workspace once; nothing in the stack reads the real extract ([ADR 0012](docs/adr/0012-api-and-compose.md)).

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
