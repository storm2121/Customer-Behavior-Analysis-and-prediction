# Milestones

Each milestone ends with its definition of done (DoD) passing and `just check` green, then an entry
in `PROGRESS.md`. The CV line under each milestone is the bullet it unlocks once real numbers exist
in the ledger; numbers in those lines are placeholders written as `{n}` and are filled only from
claim ids.

## M0. Scaffold and guard rails

- `uv` project on Python 3.12, `pyproject.toml`, `uv.lock`, `.python-version`, `justfile`,
  `.pre-commit-config.yaml` (ruff, mypy, privacy guard, nbstripout), `.editorconfig`, `.gitignore`
  covering every private artefact, `CHANGELOG.md`, `PRIVACY.md`, `docs/adr/` with ADRs 0001–0006
  filled from the stubs, `.github/workflows/ci.yml` (lint, type, test).
- Move `customer_analytics/`, `cli.py`, old `tests/` and old requirements into `legacy/` unchanged,
  with `legacy/README.md` listing the bugs from ARCHITECTURE §17.
- `kashf/` package skeleton with `config.py` (pydantic-settings), `cli.py` (typer) and
  `privacy_guard.py` working and tested.
- DoD: `uv run kashf --help`, `uv run pytest`, `uv run mypy kashf`, `pre-commit run --all-files` all
  pass; the privacy guard blocks a staged fixture that mimics an extract header.
- CV line: none yet.

## M1. Contracts, ingest, data quality, public synthetic generator

- pandera schemas for the five extracts (`kashf/contracts/`), loaders with manifest and
  reconciliation checks (`kashf ingest`, `kashf dq`), DuckDB `raw` schema.
- `kashf synth` parametric generator reproducing schema, encodings, date format and the quirks in
  ARCHITECTURE §5.1, with hypothesis invariants.
- DoD: `kashf synth --out .kashf/synth && kashf ingest --data-dir .kashf/synth && kashf dq` passes;
  the same two commands run locally against `KASHF_DATA_DIR` and the DQ report records the repeat
  rows, the zero-amount rows, the sub-1-MAD rows and the non-joining aggregate file (counts only).
- CV line: "Built a typed ingestion layer with schema contracts and reconciliation checks that
  surfaced {n} undecidable repeat rows and a silently empty join in the original analysis."

## M2. dbt warehouse

- `dbt/` project on dbt-duckdb: staging (seeds for code maps, English names), intermediate
  (`int_events`, `int_client_month`, `int_channel_state`), marts (`client_360`, `features_silence`
  parameterised by cutoff, `features_segments`, `features_anomaly`, `daily_counts`,
  `aggregate_clients`), generic and custom tests, docs generated.
- DoD: `uv run kashf dbt run && uv run kashf dbt test` green on synthetic; marts row counts match
  the synthetic manifest; `dbt docs generate` works.
- CV line: "Modelled three banking channels into a dbt warehouse on DuckDB with {n} tested models and
  window-function features for recency, rolling sums and monthly channel states."

## M3. Evidence ledger, robustness checker, findings docs

- `kashf/ledger/`: registry, `build`, `render`, `check`, allowlist, robustness checker.
- First claims: channel_mix, age_digital_divide, weekday_pattern, benford_purchases (public), and
  online_service_mix, branch_deposit_share, december_deposit_wave, salary_day_withdrawals,
  ecommerce_international_share, spend_concentration, repeat_rows, aggregate_join_coverage,
  silence_by_channel (headline, expected to reverse), ramadan_card_effect, pre_eid_branch_deposits (private).
- `docs/findings/*.md.j2` rendered; README gains a ledger-governed "What the data says" section with
  only public claims; CI runs `kashf ledger check`.
- Private report v1 (`kashf report --variant private`) written under `KASHF_PRIVATE_DIR`.
- DoD: `kashf ledger build --public`, `kashf ledger render`, `kashf ledger check` pass; a test proves
  that a hand-typed number in a governed section fails the check; the silence_by_channel claim carries
  verdict `reverses` on real data and the mechanism is explained in its caveats.
- CV line: "Designed an evidence ledger where every number in the documentation is generated from a
  registered, bootstrapped claim and CI fails on drift, plus a stratification check that showed a
  headline retention finding reversing once activity level was controlled."

## M4. Calendar, seasonality, concentration, Benford, Markov simulator

- `kashf/analysis/` modules per ARCHITECTURE §9, each producing ledger claims and figures.
- Bayesian calendar model with posterior predictive checks (fallback path documented in ADR 0007).
- Markov channel chain with Dirichlet posterior, stationary distribution, and `simulate(policy)`.
- DoD: `kashf analyse all` runs on synthetic and locally on real data; claims for each module exist;
  the simulator returns bands for a sample policy; equations documented in `docs/methods/`.
- CV line: "Fitted negative-binomial and Bayesian calendar models of daily transactions with Ramadan,
  Eid and salary-day effects, and a Dirichlet-posterior channel Markov chain that simulates branch
  load under migration policies with uncertainty bands."
- **Stop here and report** (AGENT_RULES §5.5), unless context allows continuing safely.

## M5. Risk models: silence, survival, adoption, segments, registry

- `kashf/models/` per ARCHITECTURE §10 with time-based splits and rolling-origin backtests,
  baselines, GLM odds ratios, LightGBM + Optuna, calibration, SHAP, lifelines survival, adoption
  on the aggregate file, GMM/KMeans/UMAP segments with bootstrap stability, MLflow registry, model cards.
- DoD: `kashf train all` on synthetic produces `models/registry.json` and model cards; a test asserts
  no feature uses post-cutoff data; metrics carry bootstrap CIs.
- CV line: "Predicted ninety-day client silence with time-based validation, comparing a logistic GLM
  against a tuned LightGBM with isotonic calibration (Brier {n}, lift at 10% {n}), and profiled
  segments selected by BIC with bootstrap stability {n}."

## M6. Anomaly layer and review queue

- Rules engine from YAML, Isolation Forest, empirical-Bayes amount models, Benford per client,
  combined scoring with reasons, precision at k, alert budget; Postgres schema (alerts, feedback,
  annotations) with Alembic migrations and SQLite fallback.
- DoD: `kashf anomaly run` produces a ranked queue on synthetic; feedback round-trips through the
  database; precision at k is computed against pseudo-labels; rules are unit-tested with hand-built
  cases (structuring, pass-through, velocity).
- CV line: "Built a transaction-monitoring triage layer combining configurable rules (structuring,
  velocity, pass-through) with Isolation Forest and empirical-Bayes amount models, ranked by an alert
  budget and improved by analyst feedback."

## M7. API and Compose

- FastAPI service per ARCHITECTURE §11, Prometheus metrics, OpenAPI, httpx tests; Dockerfiles;
  `docker-compose.yml` with api, postgres, dashboard placeholder, mlflow.
- DoD: `docker compose up` serves `/health` and every endpoint on synthetic data; API tests green in CI.
- CV line: "Served risk scores, segments, alerts and a policy simulator through a FastAPI service
  with PostgreSQL feedback storage, Alembic migrations and Prometheus metrics, packaged with Docker Compose."

## M8. Dashboard

- React 19 + TypeScript dashboard per ARCHITECTURE §13 with all pages, Storybook, Vitest, Playwright
  smoke, axe checks, light and dark themes, KaTeX methods page.
- DoD: `pnpm build`, `pnpm test`, `pnpm storybook:build`, Playwright smoke against Compose all pass;
  the Findings page renders claims with CI and robustness badges; the Simulator page drives
  `/markov/simulate`.
- CV line: "Designed a nine-page analytics dashboard in React and TypeScript (customer 360, segments,
  risk, alerts, what-if simulator) with Storybook-documented components and accessibility checks."

## M9. Orchestration and the private twin

- Airflow DAG `kashf_daily` running the whole pipeline in Compose on synthetic data; `airflow dags test` passes.
- `kashf synth-twin` with DP-noised copula, fidelity and privacy metrics, hard-refusing any output
  path inside the repository; run locally only.
- DoD: DAG test green; twin metrics written under `KASHF_PRIVATE_DIR`; a test asserts the refusal.
- CV line: "Orchestrated ingestion, dbt, training, evaluation and reporting as an Airflow DAG, and
  generated a differentially-private synthetic twin whose fidelity and membership-inference risk are
  measured before any demo use."

## M10. Documentation and demo

- README rewrite (what, why, how; architecture diagram; screenshots from synthetic; honesty and
  limitations; privacy; the legacy bugs), model and data cards, CHANGELOG, ADRs complete,
  `docs/CV_NOTES.md` with every bullet traced to claim ids, `just demo` one-command run verified on a
  clean clone.
- DoD: a clean clone with no real data completes `just demo`; CI green; `kashf ledger check` green.
- CV line: assembled from `docs/CV_NOTES.md`.
