# Kashf task runner. Every recipe line is a single command that works the same in PowerShell and
# in POSIX shells, so `just` behaves identically on Windows and in CI.

set windows-shell := ["powershell.exe", "-NoLogo", "-NoProfile", "-Command"]

# Tools print non-ASCII notes; a legacy Windows console code page must not crash them.
export PYTHONUTF8 := "1"

seed := "42"
clients := "1000"
synth_dir := ".kashf/synth"

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
    uv run --frozen kashf privacy-check --all-files

# Everything CI runs. A milestone is done only when this is green.
check: lint test privacy

# Write the public synthetic extract.
synth:
    uv run --frozen kashf synth --out {{synth_dir}} --clients {{clients}} --seed {{seed}}

# Load the synthetic extract into the public warehouse (contracts and reconciliation).
ingest: synth
    uv run --frozen kashf ingest --data-dir {{synth_dir}}

# Data-quality report on the public warehouse.
dq: ingest
    uv run --frozen kashf dq

# Build the public warehouse with dbt (seeds, models and every dbt test) and its documentation.
dbt: dq
    uv run --frozen kashf dbt build
    uv run --frozen kashf dbt docs generate

# Private zone: load the real extract from KASHF_DATA_DIR; outputs go to KASHF_PRIVATE_DIR only.
private-ingest:
    uv run --frozen kashf --private ingest
    uv run --frozen kashf --private dq

# Private zone: the warehouse on the real extract (dbt target and logs stay private too).
private-dbt: private-ingest
    uv run --frozen kashf --private dbt build

# Run every pre-commit hook over the whole tree.
hooks:
    uv run --frozen pre-commit run --all-files
