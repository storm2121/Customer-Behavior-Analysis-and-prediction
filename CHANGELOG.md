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

### Changed

- The 2024 script and the 2025 package moved, unchanged, into `legacy/`.
- The project was renamed from Kashf to Strata: package and CLI `strata`, variables `STRATA_*`,
  workspace `.strata/`, branch `feat/strata-rebuild`. The README and the findings pages now present
  the pipeline as the product and the findings as an illustration on the reference instance.

### Fixed

- Bootstrap intervals depended on the order in which DuckDB returned rows, which can change after
  a rebuild; warehouse reads are now in a canonical order, so the same data always gives the same
  intervals.

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
