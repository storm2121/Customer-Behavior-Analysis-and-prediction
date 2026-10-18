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
check: lint test privacy ledger-check

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

# Refresh the allow-listed public claims from the real extract and re-render the documents.
publish-claims: private-dbt
    uv run --frozen strata --private ledger build --public
    uv run --frozen strata ledger render

# Run every pre-commit hook over the whole tree.
hooks:
    uv run --frozen pre-commit run --all-files
