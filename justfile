# Kashf task runner. Every recipe line is a single command that works the same in PowerShell and
# in POSIX shells, so `just` behaves identically on Windows and in CI.

set windows-shell := ["powershell.exe", "-NoLogo", "-NoProfile", "-Command"]

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

# Run every pre-commit hook over the whole tree.
hooks:
    uv run --frozen pre-commit run --all-files
