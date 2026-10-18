# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/) from its first tagged release.

## [Unreleased]

The Strata rebuild (`strata` package), developed on branch `feat/strata-rebuild`. Strata is a
pipeline for multi-channel retail-banking event data; the CIH Bank extract is its reference
instance and the synthetic generator a first-class source.

### Added

- `uv` project on Python 3.12 with a lock file, `ruff`, `mypy --strict`, `pytest` with `hypothesis`
  and a coverage gate, `pre-commit` hooks, a `justfile` and a CI workflow.
- `strata` package skeleton: typed settings (`pydantic-settings`) that refuse private locations
  inside the repository, public and private zones, redacting structured logs, and a `typer` CLI
  that reports failures without tracebacks or identifiers.
- The privacy guard (`strata privacy-check`), run on every commit and in CI.
- `PRIVACY.md`, architecture decision records 0001 to 0006 filled in, and ADR 0008.
- Code-map seeds (`dbt/seeds/codes_*.csv`) as the single definition of every categorical value,
  with the evidence behind each assumed meaning.
- Typed pandera contracts for the five extracts; violations are reported as counts, never values.
- `strata synth`: the public parametric generator (hand-specified distributions, the extract's
  quirks and file format, planted monitoring scenarios, a hash-carrying synthetic marker).
- `strata ingest`: contract checks, the DuckDB raw layer, a manifest with a data fingerprint, and
  reconciliation between files and warehouse; public runs refuse unmarked data.
- `strata dq`: a sample-free data-quality report (exact repeats, zero and sub-one amounts,
  vocabulary, plausibility, join coverage, the aggregate file's empty rows and semantics).
- The dbt warehouse on DuckDB: five staging models, `int_events`, `int_client_month` (a dense
  client-month panel with rolling sums and recency), `int_channel_state`, and the marts
  `client_360`, `features_silence` (every cutoff of the seed, no feature past its cutoff),
  `features_segments`, `features_anomaly`, `daily_counts` (dense, with calendar columns) and
  `aggregate_clients`; generic and singular tests on every model, including a reconciliation of
  the exact-repeat flags with the data-quality report and of the marts with the manifest.
- `strata dbt …`: dbt on the active zone's warehouse, with target and logs inside that zone and
  telemetry off.
- Seeds for the age and tenure bands, the silence cutoffs and the calendar periods and holidays,
  read by both dbt and Python.
- The evidence ledger: a claim registry with client-bootstrap intervals, fifteen claims (four
  allow-listed for publication), per-value client counts, data-aware caveats and provenance;
  `strata ledger build` (synthetic, private, and `--public` through the publication rules),
  `strata ledger render` (findings documents and the README section from templates) and
  `strata ledger check` (drift, documents without templates, hand-typed numbers, allowlist).
- The robustness checker: crude, per-stratum and Mantel-Haenszel adjusted effects with intervals,
  sign flips, shrinkage and a holds / weakens / reverses / insufficient verdict on every headline.
- `strata report`: a private HTML report of every claim (private folder only) and a public demo
  report on synthetic data.
- `docs/findings/` rendered from the public claims, `docs/methods/robustness.md`.

- `.env.local` (gitignored) holds the machine's private locations; the settings read it, and the
  privacy guard learns the private locations from it and refuses to commit any `.env` file.
- An MIT licence.
- `strata analyse [all|calendar|seasonality|concentration|benford|markov]`: each analysis module
  recomputes its claims (merged into the zone's ledger), writes its figures to the zone's figures
  folder and, for the Markov chain, the simulator's inputs and the sample policy's bands.
- Calendar models: a negative-binomial regression per daily series with client-bootstrap
  intervals (`calendar_effects`) and the same model in PyMC with posterior predictive checks,
  convergence diagnostics and P(December > 2x) (`calendar_bayes`); ADR 0007 records the backend
  and ADR 0009 the terms.
- Seasonality: STL strength of the weekly cycle and of the trend per channel
  (`weekly_seasonality`); the December wave, pre-Eid week and Ramadan claims use shared, tested
  decompositions.
- Concentration: RFM quintile scores per channel with the value share of the best clients
  (`rfm_by_channel`).
- Benford: first-digit and first-two-digit tests per category (`benford_by_category`) and the
  most over-represented leading pairs of purchases (`benford_two_digits`).
- A Markov chain of monthly channel states with a Dirichlet posterior, its stationary distribution
  and mean first-passage times (`channel_markov`), and a what-if simulator with paired posterior
  paths (`markov_policy_simulation`).
- `strata ledger build --quick` and `--only` merging: model-based claims at lower effort for tests
  and demos, marked in each claim and refused by `--public`; partial builds keep the other claims
  of the same data in registry order, and a claim that is attempted and skipped loses its old
  record.
- Claim records carry an interval label (a 95% CI, a 95% credible interval, a 90% posterior
  interval), used wherever an interval is rendered.
- `docs/methods/` pages with the equations of every module: calendar, seasonality, concentration,
  Benford and Markov.
- Publication modes: `spend_concentration` is published in full and `silence_by_channel` as a
  verdict only (verdict, stratifiers and an explanation in words, never a number), enforced by the
  build, the renderer, the ledger check and the commit guard; public values on too few clients are
  withheld.
- `docs/data_card.md` (draft) with a generated table of every assumed or undocumented meaning.
- Markov simulator inputs carry state counts by age band.
- Source mappings (`mappings/*.yaml`, `strata.contracts.mapping`): `strata ingest --mapping` loads
  any extract of clients and dated channel events into the canonical tables `raw.clients`,
  `raw.events` and `raw.client_aggregates`; dbt staging reads only those. Files read through the
  reference contracts are kept in `landing.*` for the source-specific data-quality checks; other
  sources get source-agnostic checks. The manifest and the ledger record the mapping's digest.
  A snapshot test proves the reference behaviour unchanged, and a property-based test maps a
  generated alien schema onto identical marts.
- `strata train [all|silence|survival|adoption|segments] [--quick] [--promote]` and
  `strata promote` (`strata.models`): silence risk with baselines, GLM odds ratios, LightGBM
  tuned with Optuna on the earliest cutoff, isotonic and Platt calibration and SHAP reasons;
  survival of the gaps between active days and of time to silence (Kaplan-Meier with
  client-bootstrap medians, log-rank, Cox with client-clustered errors and a proportional-hazards
  test); online adoption on the aggregate sample with out-of-fold metrics and a decile table; and
  behavioural segments (Gaussian mixtures by BIC, KMeans silhouettes, bootstrap ARI stability,
  profiles with tests, generated names, UMAP for display). Every metric has a client-bootstrap
  interval; runs are logged to MLflow in the workspace; `models/registry.json` and
  `docs/model_cards/` are published from synthetic models only and checked for drift.
- A leakage test rewrites every event after a cutoff, rebuilds the features with dbt and proves
  that cutoff's features unchanged.
- The monitoring layer (`strata anomaly run|queue|feedback`, `strata.models.anomaly`): six triage
  rules configured in YAML with the evidence behind every hit, an Isolation Forest, an
  empirical-Bayes model of each client's amounts, per-client Benford tests, a combined score with
  reasons, an alert-budget selector, and precision at k against pseudo-labels (two rule families)
  and against analyst feedback. The review store (`strata.review`): runs, alerts, feedback and
  annotations in PostgreSQL or SQLite through SQLAlchemy and Alembic migrations; real alerts stay in
  a SQLite file under `STRATA_PRIVATE_DIR` (ADR 0011).
- The API (`strata.api`, `strata serve`): `/health`, `/clients/{id}/360`, `/segments`,
  `/risk/silence`, `/adoption/deciles`, `/alerts` with `POST /alerts/{id}/feedback`,
  `/calendar/effects`, `POST /markov/simulate` (age-band policies from the simulator inputs) and
  `/ledger/claims`; Prometheus metrics by route template; an optional API key; OpenAPI docs.
  `Dockerfile.api`, `Dockerfile.mlflow` and `docker-compose.yml` (API, PostgreSQL, MLflow UI and a
  dashboard placeholder) on synthetic data (ADR 0012).

### Changed

- The 2024 script and the 2025 package moved, unchanged, into `legacy/`.
- The project was renamed from Kashf to Strata: package and CLI `strata`, variables `STRATA_*`,
  workspace `.strata/`, branch `feat/strata-rebuild`. The README and the findings pages now present
  the pipeline as the product and the findings as an illustration on the reference instance.

### Fixed

- Calendar dates follow the Moroccan moon sighting (Ramadan from the twelfth of March, Eid
  al-Adha from the seventeenth of June), Islamic New Year and Mawlid are bank holidays, and every
  date is tagged `assumed` until confirmed. Rebuild an existing warehouse with
  `strata dbt build --full-refresh` once, because the seed tables gained columns.

- Bootstrap intervals depended on the order in which DuckDB returned rows, which can change after
  a rebuild; warehouse reads are now in a canonical order, so the same data always gives the same
  intervals.
- `daily_counts.is_bank_holiday` missed the two Eids, on which branches close; it now has the
  definition of `strata.holidays.is_bank_holiday` (a test keeps the two in agreement), so branch
  series no longer count Eid days as open days.

### Known defects of the legacy code

These are the reasons for the rebuild; each is explained in [`legacy/README.md`](legacy/README.md).

- SMOTE oversampling ran before the train/test split, so the reported accuracy was inflated.
- The join with the aggregate file matched no rows (it is a different sample of clients) and left
  empty columns without any warning.
- Recency was measured from the day the script ran instead of the end of the extract.
- The target, each client's most frequent card transaction type, was a weak question, and missing
  labels became a spurious `'nan'` class.
- No time-based validation and no uncertainty on any result.
- The script's sample output printed identifiers and ages of unrelated clients next to predictions.

## legacy 2025-08-18: `customer_analytics` package

The 2024 script restructured into a tested package with a CLI. It fixed the scikit-learn 1.4 crash,
the fixed-seed "random" prediction button and the unreachable manual-input window, but kept the
analytical design and its defects. Frozen in `legacy/2025_package/`.

## legacy 2024-11-04: `Analysis.py`

The original hackathon script. Frozen in `legacy/2024_script/`.
