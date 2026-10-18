# Progress log

Append one entry per milestone (or per blocked item). Never include identifiers or raw values from
the real data. Record exact commands and pass/fail counts.

## Template

### M<n>: <title> — <status: done | partial | blocked> — <date>

- Built:
- Commands run and results:
- Decisions (with ADR ids where relevant):
- Stubbed or skipped, and why:
- Open questions for the architect:
- Next:

## Entries

### Architecture — done — 2026-10-18

- Built: `ARCHITECTURE.md`, `AGENT_RULES.md`, `MILESTONES.md`, ADR stubs 0001–0006, branch `feat/kashf-rebuild`.
- Decisions: Python 3.12 via uv (ADR 0001 context), DuckDB + dbt core, evidence ledger, time-based
  validation, privacy boundary with public parametric generator and private twin, Airflow, React + TS.
- Next: M0.

### M0: Scaffold and guard rails — done — 2026-10-18

- Built:
  - `uv` project on Python 3.12 (`pyproject.toml`, `uv.lock`, `.python-version`), `.editorconfig`,
    `.gitattributes` (LF everywhere; the machine default is CRLF), `.gitignore` covering every
    private artefact type, `justfile`, `.pre-commit-config.yaml` (hygiene hooks, privacy guard,
    ruff format, ruff check, mypy strict, nbstripout; local hooks run through `uv run --frozen`),
    `.github/workflows/ci.yml` (lint and types, tests, privacy guard).
  - `legacy/`: the 2024 script restored verbatim from the first upload in git history
    (`legacy/2024_script/Analysis.py`) and the 2025 package moved unchanged with `git mv`
    (`legacy/2025_package/`); `legacy/README.md` lists fourteen bugs, including the six from
    ARCHITECTURE §17 plus a spurious `'nan'` class, misattributed sample output, mislabelled charts
    and a misnamed feature.
  - `kashf/`: `config.py` (pydantic-settings; refuses `KASHF_DATA_DIR` or `KASHF_PRIVATE_DIR`
    inside the repository, including via `..`; public and private zones), `log.py` (structlog to
    stderr; identifier fields replaced, identifier-shaped tokens masked), `cli.py` (typer; `info`,
    `privacy-check`; no tracebacks with locals; sanitised errors; private-run tracebacks written
    under the private folder), `contracts/specs.py` (the five extract shapes), `ledger/allowlist.py`
    (the four public claim ids and the minimum cell size), `privacy_guard.py` (nine rules; never
    echoes the offending token).
  - Docs: `README.md` (no numbers), `PRIVACY.md`, `CHANGELOG.md`, ADRs 0001–0006 filled from the
    stubs, ADR 0008, ADR index.
- Commands run and results:
  - `uv run kashf --help`: exit 0.
  - `uv run pytest`: 76 passed, 0 failed, 0 skipped; coverage 96.65% (gate 85%).
  - `uv run mypy` (strict on `kashf`): no issues in 9 source files.
  - `uv run ruff format --check .` and `uv run ruff check .`: clean.
  - `uv run pre-commit run --all-files`: every hook passed (the first run normalised CRLF endings in
    six files that had been edited with a Windows text writer; second run clean).
  - `just check`: exit 0.
  - Guard demonstration in this repository: staged a fixture whose header mimics the card extract;
    `uv run kashf privacy-check --staged` exited 1 with `[extract-header]`; `git commit` was refused
    by the hook (HEAD unchanged); the fixture was then removed. The same behaviour is covered by
    `tests/test_privacy_guard.py::test_staged_fixture_mimicking_an_extract_header_is_blocked`.
  - With `KASHF_DATA_DIR` set, `uv run kashf privacy-check --all-files` is clean using the
    identifier rule derived from the extract (exact-match and shape rules both active).
- Decisions:
  - ADR 0008: explicit zones; private runs write the warehouse, dbt artefacts, ledger, reports and
    logs under `KASHF_PRIVATE_DIR`; public runs will refuse data folders without the synthetic
    marker (enforcement lands in M1 with `kashf ingest`). ADR 0007 stays reserved for the Bayesian
    backend as ARCHITECTURE §9 specifies.
  - The guard's `private-path` rule skips the three architect documents (`AGENT_RULES.md`,
    `ARCHITECTURE.md`, `MILESTONES.md`), which already name the machine's private location; every
    other rule applies to them. See open question 1.
  - CSV policy is stricter than the minimum: CSVs are blocked outside `tests/fixtures/` and
    `dbt/seeds/`, and any CSV above 200 rows is blocked everywhere.
  - Public claims must have every value computed on at least the allowlist's minimum number of
    clients; the guard enforces it on `docs/ledger/*.json`.
  - Tooling: `uv` installed with the official installer into the user's local bin folder; `just`
    installed with `uv tool install rust-just` (so no Python task-runner fallback was needed);
    `justfile` uses PowerShell on Windows and keeps every recipe line shell-neutral.
  - pytest's shared temp root on this machine is an unreadable folder left by another process, so
    `tests/conftest.py` sets a project-specific temp root under the system temp folder (outside the
    repository, which several tests require).
- Stubbed or skipped, and why: CI has not run (nothing is pushed, by rule); it mirrors `just check`.
  The dbt, ledger and dashboard jobs are added in the milestones that build them.
- Open questions for the architect:
  1. `AGENT_RULES.md`, `ARCHITECTURE.md` (sections 5.1, 6, 8 and 16) and the task brief contain the
     machine-specific private location and real-data aggregates typed by hand. I did not edit the
     architect's documents. Should they be scrubbed before anything is pushed (the location replaced
     by "the folder in `KASHF_DATA_DIR`", the aggregates moved into private claims)?
  2. The repository has no licence. The README says the code is published for reading and review;
     choose a licence before publishing.
- Next: M1.

### M1: Contracts, ingest, data quality, public synthetic generator — done — 2026-10-18

- Built:
  - `dbt/seeds/codes_*.csv`: code maps for every categorical value (card categories and origins,
    online services, the branch operation labels with an assumed direction and category, client
    codes, aggregate-file flags), each meaning tagged `label`, `observed` or `assumed`. They are the
    single definition of the vocabularies: `kashf.contracts.vocab` reads them (AGENT_RULES §3.6).
  - `kashf/holidays.py`: Ramadan, the two Eids, the week before Eid al-Adha (ARCHITECTURE §9 dates)
    and the fixed national holidays, shared by the generator and the calendar models.
  - `kashf/contracts/extracts.py`: typed pandera contracts (strict, ordered columns; identifiers on
    event rows; parseable dates; non-negative amounts). Violations are summarised from pandera's
    structured errors as `(column, check, count)`; error messages are never used, because for some
    error classes pandera writes the failing values into the message.
  - `kashf/ingest/`: explicit text-to-type parsing with an independent line count, a manifest with a
    timestamp-free fingerprint, the DuckDB raw layer (with `_source_row`), reconciliation of row
    counts, distinct identifiers and amount sums between files and warehouse, and `kashf dq`, a
    sample-free data-quality report stored in `raw.dq_report`.
  - `kashf/synthetic/`: the public parametric generator and an exact-format writer (Latin-1, CRLF,
    the extract's number formatting, the aggregate file's interleaved empty rows), with a
    `SYNTHETIC.json` marker that records every file's hash. Planted structuring and pass-through
    scenarios for M6 are labelled in a truth file next to the data.
  - CLI: `kashf synth`, `kashf ingest`, `kashf dq`; `justfile`: `synth`, `ingest`, `dq`,
    `private-ingest`.
- Commands run and results:
  - `uv run kashf synth --out .kashf/synth && uv run kashf ingest --data-dir .kashf/synth && uv run kashf dq`
    (also `just dq`): all exit 0; contracts pass, 13 reconciliation checks pass, no `error` check;
    the report records the exact repeats, the zero-amount online rows, the sub-1-MAD card rows and
    the non-joining aggregate file.
  - Locally with `KASHF_DATA_DIR` set: `uv run kashf --private ingest` exit 0 (all five real files
    satisfy their contracts; 13 reconciliation checks pass; warehouse and manifest written under
    `KASHF_PRIVATE_DIR`, nothing in the working tree); `uv run kashf --private dq` exit 0 with no
    `error` check. The four checks named in the DoD are recorded (counts kept in the private
    warehouse, not reproduced here).
  - `uv run pytest`: 170 passed, 1 skipped (the `private` test); coverage 97.94% (gate 85%).
  - `uv run pytest --run-private -m private` (local, real data): 1 passed (asserts exit codes only).
  - `uv run mypy` (strict, 21 files) and `uv run ruff check .`: clean. `just check`: exit 0. Every
    commit went through the pre-commit hooks and `kashf privacy-check --staged`.
- Decisions:
  - Contract and data quality are separate layers: the contract stops the ingest when the
    pipeline's assumptions fail; data quality measures everything that may legitimately vary.
  - Fail-closed public zone: `kashf ingest` refuses a folder without the synthetic marker, and a
    marked folder whose files no longer match the recorded hashes.
  - The raw layer keeps the aggregate file as loaded (including its empty rows); staging filters
    them in M2. The exact-repeat definition lives in `kashf.ingest.dq.REPEAT_KEYS`; dbt staging
    will flag the same rows and a dbt test will reconcile the two counts.
  - Generator parameters are hand-set. Where ARCHITECTURE.md states a property in public (channel
    coverage, repeat share, card-service zero rows, sub-1-MAD rows, heavy branch tail, round
    withdrawals, December wave, salary days, international e-commerce) it is reused, rounded; every
    other parameter is a round, plausible value, not fitted to the extract.
  - The silence-by-channel pattern is produced by mechanisms rather than imposed: habitual branch
    visits are regular (gamma-renewal gaps), and digital dormancy falls with digital engagement.
    Before adopting the defaults I checked five seeds: each shows a positive crude gap between
    branch-only and digital-only silence and a negative gap within activity terciles.
  - Synthetic identifiers are `syn_` plus twelve hex characters, so they can never match the
    identifier rule of the guard.
  - `PYTHONUTF8=1` in the `justfile` and CI: mypy's notes contain non-ASCII characters and crashed
    once on the Windows console code page (not a type error).
- Stubbed or skipped, and why: nothing stubbed. The planted monitoring scenarios are unused until
  M6.
- Open questions for the architect:
  1. The directions and categories of the rare branch labels are assumptions (`semantics=assumed`
     in the seed); the bank's documentation would settle them.
  2. The meaning of several aggregate-file codes (`civilite`, `flag_debit_credit`,
     `flag_contrat_actif`) is undocumented and kept as categories.
- Next: M2.
