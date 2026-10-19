# Strata task runner. Every recipe line is a single command that works the same in PowerShell and
# in POSIX shells, so `just` behaves identically on Windows and in CI.

set windows-shell := ["powershell.exe", "-NoLogo", "-NoProfile", "-Command"]

# Tools print non-ASCII notes; a legacy Windows console code page must not crash them.
export PYTHONUTF8 := "1"

seed := "42"
clients := "1000"
synth_dir := ".strata/synth"

# List the recipes.
default:
    @just --list

# Create the environment and install the git hooks.
setup:
    uv sync --frozen
    uv run --frozen pre-commit install

# Format the code base.
fmt:
    uv run --frozen ruff format .
    uv run --frozen ruff check --fix .

# Lint and type-check without changing anything.
lint:
    uv run --frozen ruff format --check .
    uv run --frozen ruff check .
    uv run --frozen mypy

# Unit and integration tests on synthetic data, with the coverage gate.
test:
    uv run --frozen pytest

# Scan every tracked file with the privacy guard.
privacy:
    uv run --frozen strata privacy-check --all-files

# The documents say exactly what the public ledger says (drift, hand-typed numbers, allowlist).
ledger-check:
    uv run --frozen strata ledger check

# Everything CI runs. A milestone is done only when this is green.
check: lint test privacy ledger-check dashboard-check

# Write the public synthetic extract.
synth:
    uv run --frozen strata synth --out {{synth_dir}} --clients {{clients}} --seed {{seed}}

# Load the synthetic extract into the public warehouse (contracts and reconciliation).
ingest: synth
    uv run --frozen strata ingest --data-dir {{synth_dir}}

# Data-quality report on the public warehouse.
dq: ingest
    uv run --frozen strata dq

# Build the public warehouse with dbt (seeds, models and every dbt test) and its documentation.
dbt: dq
    uv run --frozen strata dbt build
    uv run --frozen strata dbt docs generate

# Every claim on the synthetic warehouse, and the public demo report.
ledger: dbt
    uv run --frozen strata ledger build
    uv run --frozen strata report --variant public

# Analysis modules on the synthetic warehouse: claims, figures and the simulator (quick effort).
analyse: dbt
    uv run --frozen strata analyse all --quick

# Risk models on the synthetic warehouse, then the public registry and model cards.
train: dbt
    uv run --frozen strata train all --promote

# The monitoring layer on the synthetic warehouse: the ranked queue, stored for review.
anomaly: dbt
    uv run --frozen strata anomaly run --budget 50

# Serve the API on the public workspace (http://127.0.0.1:8000/docs).
serve:
    uv run --frozen strata serve

# The dashboard recipes run inside apps/dashboard, so corepack uses the pnpm that its package.json
# pins (run from the repository root, it would pick a version of its own).

# The dashboard in development (http://localhost:5173), proxying /api to `just serve`.
[working-directory: 'apps/dashboard']
dashboard:
    corepack pnpm install --frozen-lockfile
    corepack pnpm dev

# The dashboard's checks: types, unit tests, the production build with its bundle check, Storybook.
[working-directory: 'apps/dashboard']
dashboard-check:
    corepack pnpm install --frozen-lockfile
    corepack pnpm typecheck
    corepack pnpm test
    corepack pnpm build
    corepack pnpm storybook:build

# Set STRATA_DASHBOARD_URL when the dashboard port is moved and STRATA_E2E_CHANNEL=msedge (or
# chrome) to drive an installed browser instead of one downloaded by `pnpm exec playwright install`.
# Playwright smoke tests with axe checks against the running stack (`just up`).
[working-directory: 'apps/dashboard']
dashboard-e2e:
    corepack pnpm e2e

# Port overrides of this machine (STRATA_*_PORT) are read from .env.local when it exists.
compose_env := if path_exists(".env.local") == "true" { "--env-file .env.local" } else { "" }

# The whole stack on synthetic data in the foreground: API, PostgreSQL, MLflow UI and the dashboard.
up:
    docker compose {{compose_env}} up --build

# The extract, the warehouse, every claim, the analysis modules, the models and their gate, the
# monitoring queue and the demo report, then the stack in the background and a smoke test of every
# endpoint and of the dashboard. Nothing in it reads the real extract or changes a tracked file.
# One command on synthetic data, from a clean clone; `just down` stops the stack.
demo: dbt
    uv run --frozen strata ledger build --quick
    uv run --frozen strata analyse all --quick
    uv run --frozen strata train all --quick
    uv run --frozen strata evaluate
    uv run --frozen strata anomaly run --budget 50
    uv run --frozen strata report --variant public
    docker compose {{compose_env}} up --build --detach
    uv run --frozen python docker/smoke.py --compose

# Stop the stack and remove its volumes.
down:
    docker compose {{compose_env}} --profile orchestration down --volumes

# The daily DAG end to end in its Compose image on synthetic data (`airflow dags test`).
dag-test:
    docker compose {{compose_env}} --profile orchestration build airflow
    docker compose {{compose_env}} --profile orchestration run --rm airflow dags test strata_daily

# Private zone: load the real extract from STRATA_DATA_DIR; outputs go to STRATA_PRIVATE_DIR only.
private-ingest:
    uv run --frozen strata --private ingest
    uv run --frozen strata --private dq

# Private zone: the warehouse on the real extract (dbt target and logs stay private too).
private-dbt: private-ingest
    uv run --frozen strata --private dbt build

# Private zone: every claim on the real extract and the private report (private folder only).
private-report: private-dbt
    uv run --frozen strata --private ledger build
    uv run --frozen strata --private report --variant private

# Private zone: analysis modules at full effort on the real extract (private folder only).
private-analyse: private-dbt
    uv run --frozen strata --private analyse all

# Private zone: risk models on the real extract (registry, cards and scores stay private).
private-train: private-dbt
    uv run --frozen strata --private train all

# Refresh the allow-listed public claims from the real extract and re-render the documents.
publish-claims: private-dbt
    uv run --frozen strata --private ledger build --public
    uv run --frozen strata ledger render

# Run every pre-commit hook over the whole tree.
hooks:
    uv run --frozen pre-commit run --all-files
