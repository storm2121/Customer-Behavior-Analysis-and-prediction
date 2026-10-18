# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/) from its first tagged release.

## [Unreleased]

The Kashf rebuild (`kashf` package), developed on branch `feat/kashf-rebuild`.

### Added

- `uv` project on Python 3.12 with a lock file, `ruff`, `mypy --strict`, `pytest` with `hypothesis`
  and a coverage gate, `pre-commit` hooks, a `justfile` and a CI workflow.
- `kashf` package skeleton: typed settings (`pydantic-settings`) that refuse private locations
  inside the repository, public and private zones, redacting structured logs, and a `typer` CLI
  that reports failures without tracebacks or identifiers.
- The privacy guard (`kashf privacy-check`), run on every commit and in CI.
- `PRIVACY.md`, architecture decision records 0001 to 0006 filled in, and ADR 0008.
- Code-map seeds (`dbt/seeds/codes_*.csv`) as the single definition of every categorical value,
  with the evidence behind each assumed meaning.
- Typed pandera contracts for the five extracts; violations are reported as counts, never values.
- `kashf synth`: the public parametric generator (hand-specified distributions, the extract's
  quirks and file format, planted monitoring scenarios, a hash-carrying synthetic marker).
- `kashf ingest`: contract checks, the DuckDB raw layer, a manifest with a data fingerprint, and
  reconciliation between files and warehouse; public runs refuse unmarked data.
- `kashf dq`: a sample-free data-quality report (exact repeats, zero and sub-one amounts,
  vocabulary, plausibility, join coverage, the aggregate file's empty rows and semantics).
- The dbt warehouse on DuckDB: five staging models, `int_events`, `int_client_month` (a dense
  client-month panel with rolling sums and recency), `int_channel_state`, and the marts
  `client_360`, `features_silence` (every cutoff of the seed, no feature past its cutoff),
  `features_segments`, `features_anomaly`, `daily_counts` (dense, with calendar columns) and
  `aggregate_clients`; generic and singular tests on every model, including a reconciliation of
  the exact-repeat flags with the data-quality report and of the marts with the manifest.
- `kashf dbt …`: dbt on the active zone's warehouse, with target and logs inside that zone and
  telemetry off.
- Seeds for the age and tenure bands, the silence cutoffs and the calendar periods and holidays,
  read by both dbt and Python.

### Changed

- The 2024 script and the 2025 package moved, unchanged, into `legacy/`.

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
