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

### M2: dbt warehouse — done — 2026-10-18

- Built:
  - `dbt/` (dbt-core 1.12, dbt-duckdb 1.11): staging for the five extracts (English names, code maps
    joined from seeds, exact-repeat and zero-amount flags kept as columns, the aggregate file's empty
    rows dropped); `int_events` (the union of the three channels), `int_client_month` (a dense panel
    of every client in every month with rolling three-month sums, events to date and months since
    the last active month), `int_channel_state` (none / branch_only / digital_only / mixed, with the
    previous month's state); marts `client_360`, `features_silence` (one row per cutoff of the
    `silence_cutoffs` seed and client, features from events on or before the cutoff only),
    `features_segments`, `features_anomaly` (deposit velocity, sub-threshold clusters, pass-through,
    cash-to-card ratio, dormant-then-burst; thresholds are dbt variables), `daily_counts` (dense, with
    day of week, five-day month bins, month, trend, holidays, Ramadan, the Eids and the pre-Eid week)
    and `aggregate_clients` (the second sample with the two adoption targets).
  - Tests on every model and seed: unique, not_null, accepted_values, relationships, and custom
    generic tests (`non_negative`, `within_window`, `unique_combination`, `expression_is_true`,
    `one_row_per_client_and_month` for the state partition); singular tests reconciling the
    exact-repeat flags with the data-quality report, the models with the manifest, and a warn-level
    check that the aggregate sample shares no identifier with the client file.
  - New seeds read by both dbt and Python: age and tenure bands, the silence cutoffs with their
    outcome windows, the calendar periods and the bank holidays (`kashf/holidays.py` now builds its
    periods from them).
  - `kashf dbt …` (`kashf/dbt_runner.py`): dbt in-process against the zone's warehouse, with target
    and logs inside the zone and telemetry off; `justfile`: `dbt`, `private-dbt`; CI job `warehouse`.
- Commands run and results:
  - From an empty `.kashf/`: `uv run kashf synth --out .kashf/synth`, `uv run kashf ingest --data-dir .kashf/synth`,
    `uv run kashf dq`, then `uv run kashf dbt seed` (PASS=12), `uv run kashf dbt run` (PASS=14),
    `uv run kashf dbt test` (PASS=160, WARN=0, ERROR=0), `uv run kashf dbt docs generate` (catalog
    written). `uv run kashf dbt build`: PASS=186, WARN=0, ERROR=0.
  - Marts against the synthetic manifest (asserted in `tests/test_dbt.py` and in the singular dbt
    test): `client_360` has one row per client, `int_events` holds every event row, the aggregate
    mart every non-empty aggregate row, the channel states partition the clients in all thirteen
    months, and no feature row has a last event after its cutoff.
  - Locally on the real extract: `uv run kashf --private dq` exit 0, `uv run kashf --private dbt build`
    PASS=186, WARN=0, ERROR=0; dbt target and logs written under `KASHF_PRIVATE_DIR`; nothing new in
    the working tree.
  - `uv run pytest`: 180 passed, 1 skipped (private); coverage 97.76%. `uv run mypy` (22 files) and
    `uv run ruff check .`: clean. `just check`: exit 0.
- Decisions:
  - `features_silence` is "parameterised by cutoff" through the seed rather than a dbt variable: one
    model holds every cutoff, so the rolling-origin backtests of M5 read one table.
  - Day-of-month bins, bands, cutoffs and calendar periods are each defined once (SQL or seed), and
    the duplicate Python helper for the bins was removed (AGENT_RULES §3.6).
  - The public zone's dbt target and logs moved to `.kashf/dbt/` (the private zone's were already
    under `KASHF_PRIVATE_DIR`), so tests and demo runs never write into `dbt/`.
  - dbt-duckdb keeps a DuckDB handle open after a command and its `close_all_connections` only
    drops the reference; `release_duckdb()` closes it explicitly so the same process can open the
    warehouse read-only afterwards (needed by the ledger and the API). It touches an adapter
    internal and tolerates a change in its shape.
  - No dbt packages (no network dependency at build time); the few generic tests needed are local
    macros.
- Stubbed or skipped, and why: nothing stubbed. On private runs dbt's own log lines show the
  private folder's path in the terminal (documented in PRIVACY.md; nothing is written to the repo).
- Open questions for the architect: none new.
- Next: M3.

### M3: Evidence ledger, robustness checker, findings docs — done — 2026-10-18

The project was renamed from Kashf to Strata during this milestone (see "Architect changes applied"
below). Entries above this one use the old names (`kashf`, `KASHF_*`, `.kashf/`) as they were run.

- Built:
  - `strata/ledger/`: the `@claim` registry (a headline cannot be registered without a robustness
    specification), the client bootstrap (multinomial weights, one thousand resamples, one seed per
    claim derived from its id), claim records (values with units, intervals, clients behind every
    value, data-aware caveats, metadata, fingerprint, version), the build (synthetic, private, and
    `--public` through the publication rules), rendering, and the check.
  - Fifteen claims: the four allow-listed ones (`channel_mix`, `age_digital_divide`,
    `weekday_pattern`, `benford_purchases`) and `online_service_mix`, `branch_deposit_share`,
    `december_deposit_wave`, `salary_day_withdrawals`, `ecommerce_international_share`,
    `spend_concentration`, `repeat_rows`, `aggregate_join_coverage`, `silence_by_channel`
    (headline), `ramadan_card_effect`, `pre_eid_branch_deposits`.
  - The robustness checker (`strata/ledger/robustness.py`, method in `docs/methods/robustness.md`):
    crude, per-stratum and Mantel-Haenszel adjusted effects with intervals, sufficient strata (both
    groups at least ten clients), sign flips, shrinkage, verdict holds / weakens / reverses /
    insufficient, and a generated explanation. The rule was written and tested on constructed cases
    before it was run on any data.
  - `docs/findings/*.md.j2` rendered to `docs/findings/*.md`; the README's ledger-governed section;
    `docs/ledger/claims.public.json` (allow-listed claims only, small strata suppressed).
  - `strata report --variant private|public`: self-contained HTML (Plotly embedded). Private report
    v1 written to `<STRATA_PRIVATE_DIR>/reports/private_report.html`; the public demo report to
    `.strata/reports/`.
  - `analysis/benford.py` and `analysis/concentration.py` started here because claims need them
    (first and first-two digit tests with Nigrini bands; Lorenz, Gini, top shares, all weightable).
- Commands run and results (final runs, after the rename):
  - `uv run strata --private ledger build --public`: fifteen claims, none skipped;
    `silence_by_channel` verdict **reverses** (the reversal comes from the activity-tercile
    stratification) and `age_digital_divide` verdict holds; the public file was written after the
    publication rules passed. The mechanism is explained in the claim's robustness explanation and
    caveats: branch-only clients have far fewer events in the observation window, so a quarter
    without any event is likely by chance; activity terciles that lack one group are listed as
    positivity gaps and excluded from the adjusted difference.
  - `uv run strata ledger render` then `uv run strata ledger check`: documents match the ledger, no
    hand-typed numbers.
  - The hand-typed-number DoD is proven by three tests in `tests/test_render_check.py` (a number
    typed into a findings template, into the README template, and into the README section): each
    makes the check fail as it should, and the tests pass.
  - `uv run strata --private report --variant private`: written under `STRATA_PRIVATE_DIR`.
  - Values, intervals and robustness results are identical across two complete private rebuilds
    (ingest, data quality, dbt, ledger) after the determinism fix below.
  - `uv run pytest`: 277 passed, 1 skipped (private); coverage 97.68%. `uv run pytest --run-private
    -m private`: 1 passed. `uv run mypy` (42 files) and `uv run ruff check .`: clean. `just check`
    (lint, types, tests, privacy scan with the identifier rule derived from the extract, ledger
    check): exit 0. `uv run pre-commit run --all-files`: every hook passed.
- Decisions:
  - Headline robustness: `silence_by_channel` is checked within activity terciles, age bands and
    tenure bands; `age_digital_divide` (clients aged 55 and over against under 35, branch-only
    share) within gender codes and tenure bands only, because activity is partly a consequence of
    channel choice and holding it fixed would bias that comparison (stated in the claim's caveats).
  - Publication rules are defined once (`strata/ledger/allowlist.py`) and enforced by both the
    build and the commit guard: every value on at least the floor of clients, the compared groups
    each at the floor, and a stratum keeps its numbers only if each of its groups reaches the floor.
  - Templates may not type a digit in prose. Row labels (bands, digits, weekdays) come from claim
    metadata; `raw()` exists for conditions only, so interpretive sentences follow the data (a
    sentence about an excess of leading nines is only rendered when there is one).
  - `--public` refuses synthetic data, so the public file can only ever hold real aggregates.
  - Determinism fix: diffing a regenerated public file against the committed one showed identical
    values but different intervals. Claims read tables without an order, DuckDB's scan order can
    change between builds, and the order decided which bootstrap weight went to which client.
    Warehouse reads are now sorted on all columns, and a test proves the same rows stored in
    different orders read back identically.
- Architect changes applied at the end of this milestone:
  1. Pushing. I did not run `git push` in any form during this session. The remote-tracking reflog
     records "update by push" from this clone at 12:26:17 UTC; at that time I was writing the ledger
     render and check modules with file tools and ran no git command, so I cannot say which process
     pushed. I understand the rule: never push, never set an upstream, never run `git push` with any
     option (including `--dry-run` and `--delete`). As instructed, `git branch --unset-upstream` was
     run; the branch has no upstream.
  2. Rename to Strata: package and CLI `strata`, variables `STRATA_*`, workspace `.strata/` (with
     `.kashf/` still gitignored for old local copies), dbt project and profile `strata`, recipes, CI,
     pre-commit, README, PRIVACY, CHANGELOG, ADRs, methods, tests, the private warehouse file
     (`strata.duckdb`; the old private file was removed from `STRATA_PRIVATE_DIR`). The branch was
     renamed locally with `git branch -m feat/strata-rebuild`. The three architect documents were
     not edited.
  3. Framing: README and findings pages rewritten so that the pipeline is the product and the CIH
     findings illustrate it; ARCHITECTURE section 5.0 and milestone M4.5 read (M4.5 comes after my
     stopping point and is not started).
  4. `.env.local`: gitignored explicitly; `strata.config.Settings` reads `.env` and `.env.local`
     from the repository root (real environment variables still win), so private runs and the
     private tests need no export; the guard learns the private locations from the same settings
     and refuses any `.env` file (`.env.example` excepted); tests switch the files off and a test
     proves the suite cannot see this machine's configuration. Licence: MIT, with Mohamed as
     copyright holder (`LICENSE`); this is a decision Mohamed can reverse.
- Stubbed or skipped, and why: PDF output of the report (WeasyPrint) is not produced; the report is
  HTML only. The calendar claims are simple day-of-week standardised ratios; the regression models
  come in M4.
- Open questions for the architect:
  1. `silence_by_channel` (verdict and explanation, no numbers) is private. Should its verdict be
     allow-listed so the public documents and the CV can cite the reversal? Today nothing about it
     is published.
  2. The README's governed section now shows the four allow-listed real aggregates. Please confirm
     the allowlist is final before anything leaves the machine.
- Next: M4.

### M4: Calendar, seasonality, concentration, Benford, Markov simulator — done — 2026-10-18

- Built:
  - `strata/analysis/calendar.py`: one negative-binomial (NB2) regression per daily series (all card
    events, card withdrawals, card deposits, branch deposits on open days, online operations) on
    day of week, five-day bins of the month, sum-coded month and the Ramadan, Eid, pre-Eid and
    national-holiday terms; client-bootstrap intervals from refits at the fitted dispersion.
  - `strata/analysis/calendar_bayes.py`: the same model in PyMC with weakly informative priors,
    posterior medians and credible intervals, P(December card deposits > 2×), posterior predictive
    coverage and a standard-deviation check, R-hat, bulk ESS and divergences.
  - `strata/analysis/seasonality.py`: STL strength of the weekly cycle and of the trend, the
    December breadth and depth decomposition, the weekday-standardised ratio (the M3 claims now use
    these shared, tested functions).
  - `strata/analysis/markov.py`: transition counts between complete months, Dirichlet(1) posterior
    rows, stationary distribution, mean first-passage times, policies (share, filter, target
    state), paired posterior paths and 5/50/95% bands of branch load, digital share and silent
    share; the RFM quintile scores and per-client digit matrices for the other modules.
  - Eight new claims, twenty-three in all: `calendar_effects`, `calendar_bayes`,
    `weekly_seasonality`, `rfm_by_channel`, `benford_by_category` (first and first-two digits per
    category), `benford_two_digits`, `channel_markov`, `markov_policy_simulation`.
  - `strata analyse [all|calendar|seasonality|concentration|benford|markov] [--quick]`: recomputes
    each module's claims (merged into the zone's ledger), writes Plotly figures to the zone's
    figures folder (calendar forest plot, STL per channel, Lorenz curves, leading digits, the
    policy fan chart), prints the sample policy's bands against the baseline, and saves the
    simulator's inputs and bands (`analysis/markov_simulator.json` in the zone's workspace) for the
    API of M7.
  - Ledger: an interval label on every claim (95% CI, 95% credible interval, 90% posterior
    interval) used wherever an interval is rendered; `--quick` effort for model claims, marked in
    every claim and refused by `--public`; `--only` builds merge into the ledger of the same data
    in registry order, refuse unknown ids, and drop the old record of a claim that is attempted
    and skipped.
  - `docs/methods/` pages with the equations of every module (calendar, seasonality,
    concentration, Benford, Markov); ADR 0007 and ADR 0009; `just analyse` and
    `just private-analyse`; CI runs `strata analyse all --quick` after the ledger build.
- Commands run and results:
  - `uv run strata dbt build` (synthetic) and `uv run strata --private dbt build`, after the
    `daily_counts` fix below: PASS=186, WARN=0, ERROR=0 on both.
  - `uv run strata analyse all --quick` (synthetic): exit 0; every module's claims computed, none
    skipped; seven figures and the simulator file written under `.strata/`; the sample policy's
    bands printed for months zero, one, three, six and twelve (DoD: the simulator returns bands for
    a sample policy).
  - `uv run strata analyse all` (synthetic, full effort): exit 0 in about seventy seconds.
  - `uv run strata --private analyse all` (full effort): exit 0 in under eighty seconds; all
    fifteen module claims present at full effort, none skipped; seven figures and the simulator
    file written under `STRATA_PRIVATE_DIR`. The command's output went to the private log folder;
    I read only the exit code, the claim ids present and the file names.
  - `uv run strata --private ledger build --public`: twenty-three claims, none skipped. The public
    file's values, intervals and robustness results are identical to the committed ones; only
    `computed_at` and the new `interval_label` field changed, and that refresh is committed.
  - `uv run strata ledger render` and `uv run strata ledger check`: documents unchanged and in
    agreement with the ledger, no hand-typed numbers.
  - `uv run strata --private report --variant private`: the private report, now with all
    twenty-three claims (none at quick effort), rewritten under `STRATA_PRIVATE_DIR`.
  - `just check`: exit 0 (ruff format and check, mypy on 52 files, pytest 330 passed and 1 skipped
    with coverage 97.37%, privacy scan clean with the identifier rule derived from the extract,
    ledger check). `uv run pytest --run-private -m private`: 1 passed.
    `uv run pre-commit run --all-files`: every hook passed.
- Decisions (with ADR ids where relevant):
  - ADR 0007: PyMC 6 with PyTensor's Python backend (no C++ compiler on this machine; it installed
    in minutes, so the NumPyro and Laplace fallbacks were not needed). NUTS with one process, seeded
    by the master seed; full effort four chains of a thousand draws after a thousand tuning steps,
    quick effort two chains of three hundred.
  - ADR 0009 (a deviation from ARCHITECTURE section 9): no separate trend term, because one year
    cannot separate month from trend and the trend coefficient would only measure drift within
    months; month effects are against the average month and carry that caveat. The bank-holiday
    term covers the national holidays and the Eids keep their own terms, so every holiday effect is
    read against ordinary days. Branch series are modelled on open days.
  - Fix: `daily_counts.is_bank_holiday` left out the two Eids, so branch series counted them as
    open days and the Eid terms of the branch model were degenerate. It now has the definition of
    `strata.holidays.is_bank_holiday`, and a test keeps the mart and the Python function in
    agreement day by day.
  - Model claims use the client bootstrap like every other claim: daily series are rebuilt from
    resampled clients and refitted (two hundred refits at full effort), because days share their
    clients and are not independent units.
  - Simulator: policy and baseline are projected with the same posterior draw, so their difference
    is exact draw by draw; the claim reports differences at months zero, one, three and twelve and
    the branch operations saved over the horizon, plus the branch operations per client-month of
    each state. The model can show branch load rising under a move to mixed use, if mixed-use
    clients make more branch operations than branch-only clients; the methods page and the claim's
    caveat say so rather than hiding it.
  - The sample policy is thirty per cent of branch-only clients under 45 moving to mixed use (the
    age filter is the ARCHITECTURE example; the share is my choice). A policy may also target
    digital-only use or silence.
  - RFM scores give tied values one score (average ranks), and the bootstrap holds the quintile
    cut-offs fixed (stated in the claim). Two-digit Benford tests run per category only where there
    are at least a thousand amounts.
  - Figures are HTML pages sharing one `plotly.min.js` per folder; a private run writes them, the
    simulator file and the claims under `STRATA_PRIVATE_DIR` only, and refuses any other location.
- Stubbed or skipped, and why:
  - No NumPyro or Laplace backend: PyMC installed cleanly, so ARCHITECTURE's fallbacks were not
    triggered.
  - No posterior plots for the Bayesian model; its diagnostics are claim values instead. One figure
    per module (STL for each channel).
  - The simulator projects expected counts, so its bands carry the uncertainty of the transition
    matrix only. The saved file holds state-level aggregates, enough for the sample policy and for
    any share; filters other than the sample's will need state counts by age band, to be decided
    with the API in M7.
  - CI has not run on these commits because nothing is pushed (as instructed). The report is still
    HTML only (no PDF). M4.5 is after my stopping point and is not started.
- Open questions for the architect:
  1. The bank-holiday seed has the fixed national holidays and the code adds the two Eids; Islamic
     New Year and the Prophet's birthday (Mawlid), also public holidays in Morocco, are not in it
     because their dates follow the official moon-sighting announcement, which I could not verify
     offline. Please confirm the dates in the window to add them (they change open days and the
     holiday term). The same applies to the Ramadan and Eid dates already in the seed.
  2. Should any M4 claim be allow-listed (for example the weekly cycle or the calendar rate
     ratios)? All of them are private today.
  3. Is the sample policy (thirty per cent, under 45, to mixed use) the one to showcase, or is there
     a policy the bank would rather see simulated?
  4. Is a second year of the extract available? It would separate month from trend (ADR 0009) and
     let the chain's matrix vary by season.
  5. Still open from M3: allow-listing the `silence_by_channel` verdict, and confirming that the
     allowlist is final.
- Next: stop and report (MILESTONES M4); M4.5 (source mappings) when the architect says so.

### M4 review: the architect's answers applied — done — 2026-10-18

- Built:
  - Calendar seeds with the Moroccan dates the architect gave (Ramadan, Eid al-Fitr, Eid al-Adha
    and the week before it relative to the Eid's first day, Islamic New Year, Mawlid, the fixed
    national holidays), every row tagged `assumed` with its evidence. The bank-holiday term of the
    calendar model now covers every bank holiday other than the Eids.
  - Publication modes in `strata/ledger/allowlist.py`: `spend_concentration` published in full and
    `silence_by_channel` as `verdict_only` (verdict, stratifier names, reversing stratifiers and a
    number-free explanation generated by the robustness checker). The build writes verdicts to a
    separate `verdicts` list, the renderer answers only verdict placeholders for them, and
    `ledger_violations` (build, ledger check, commit guard) refuses any other field or a digit in
    their words. Public values computed on fewer clients than the floor are withheld with their
    table rows.
  - Every calendar claim carries the month and trend limitation; the calendar findings page states
    it too.
  - State counts by age band in the Markov simulator inputs (`counts_by_band`, `banded_start`).
  - `docs/data_card.md`: a draft whose table of every assumed or undocumented meaning is generated
    from the seeds (`python -m strata.contracts.data_card`), with a test that fails on drift.
- Commands run and results:
  - `uv run strata --private dbt build --full-refresh`: PASS=192, WARN=0, ERROR=0. A build without
    `--full-refresh` fails on the two calendar seeds, because an existing seed table keeps its old
    columns; fresh warehouses (CI, tests) are not affected. The public warehouse was rebuilt the
    same way.
  - `uv run strata --private ledger build --public`: twenty-three claims, none skipped; the public
    file now holds five full claims and one verdict; render and check pass.
  - Before and after the calendar change (comparison by kind, no values seen): only
    `calendar_effects`, `calendar_bayes`, `ramadan_card_effect` and `pre_eid_branch_deposits`
    moved; their values and intervals all changed, as expected when the dates move. No robustness
    verdict moved and no public value moved. Five frequentist rate ratios changed direction (Eid
    al-Adha for card deposits, card withdrawals and online operations; the pre-Eid week for card
    deposits; the bank-holiday term for all card events) and four changed whether their interval
    contains one (Ramadan for branch deposits and for all card events, Eid al-Adha for card
    deposits, the pre-Eid week for online operations); in the Bayesian model one direction (pre-Eid
    week, card deposits) and one interval (Ramadan, withdrawals). Every one of these terms rests on a
    handful of days, which is why the dates stay tagged until Mohamed confirms them.
  - `just check`: exit 0 (347 passed, 1 skipped, coverage 97.48%).
- Decisions:
  - Amazigh New Year (14 January) is kept in the holiday seed although it is not in the architect's
    list: it has been an official holiday since 2024. It is tagged `assumed` with that note, and it
    falls on a Sunday in the window, so it changes no open day.
  - `spend_concentration` reports the number of clients in each top group as the clients behind its
    top shares, so the top one per cent of spenders is withheld from publication while the top tenth
    and fifth and the Gini are published. This is stricter than the architect asked.
  - Anything under `docs/ledger/` that is not a ledger object now fails the guard (it used to pass a
    bare JSON string).
- Open questions for the architect: confirm Amazigh New Year; the rest stands as answered.
- Next: M4.5.

### M4.5: Source mappings (the pipeline is the product) — done — 2026-10-18

- Built:
  - `strata/contracts/mapping.py`: strict pydantic models of a mapping (defaults, file roles,
    column renames, code maps for channels, kinds and directions, per-file encoding, delimiter,
    decimal mark and date format), the canonical column sets, and the translation of a parsed file
    into canonical rows. `mappings/reference_cih.yaml` and `mappings/synthetic.yaml` (the same
    format; generated from the contracts so they cannot disagree), and `mappings/README.md`.
  - `strata ingest --mapping <file>`: files with a reference contract are read through it exactly
    as in M1 and kept in `landing.*`; other files are read generically (every mapped column present
    and parseable, row count equal to line count). Canonical `raw.clients`, `raw.events` and
    `raw.client_aggregates` are written for every source; reconciliation covers both layers.
  - dbt staging reads only the canonical tables; the manifest reconciliation test works from the
    files' roles, so it holds for any source.
  - `strata dq`: the M1 checks on the landing tables, and source-agnostic checks on the canonical
    tables for channels without one.
  - Provenance: the manifest and the ledger record the mapping's name and digest; the fingerprint
    still names the bytes; partial ledger builds merge only within the same bytes and mapping.
  - ADR 0010 (a `currency` column in canonical events, landing tables, provenance, code maps).
- Commands run and results:
  - DoD, synthetic: `tests/test_mapping.py::test_the_reference_mapping_reproduces_the_pipeline_before_mappings`
    compares the warehouse built through `mappings/synthetic.yaml` with a snapshot taken from the
    pipeline before this milestone (`tests/fixtures/reference_snapshot.json`): manifest files and
    fingerprint, data-quality report and the row count of every staging, intermediate and mart
    model are identical. Passes.
  - DoD, real data (private, compared by script, nothing printed but identical or different):
    `strata --private ingest --mapping reference_cih`, `strata --private dq`,
    `strata --private dbt build --full-refresh` (PASS=192): fingerprint, files, data-quality report
    and every model's row count identical to the snapshot taken before the change; a full
    `strata --private ledger build --public` afterwards leaves all twenty-three claims unchanged
    (values, intervals, verdicts).
  - Portability: `tests/test_mapping.py::test_an_alien_schema_maps_onto_the_same_marts`, a
    hypothesis test (three derandomized examples) that draws a schema with other column names,
    delimiter, decimal mark, date format, encoding, recoded channels, kinds and directions,
    shuffled rows and either one file per channel or one file with a channel column; the
    canonical tables and every mart are identical to the reference's. Passes.
  - `just check`: exit 0 (360 passed, 1 skipped, coverage 97.63%).
- Decisions: ADR 0010. The mapping files are looked up under the configured repository and then
  under the checkout the code runs from, so a private run from another folder still finds them.
- Stubbed or skipped, and why: the alien-schema test runs three examples, because each builds a
  full warehouse; the schema space it samples is listed in `tests/alien.py`.
- Open questions for the architect: none.
- Next: M5.

### M5: Risk models: silence, survival, adoption, segments, registry — done — 2026-10-18

- Built (`strata/models/`):
  - `splits`: the cutoffs of the `silence_cutoffs` seed, backtests that train on one cutoff and
    test on the next (the primary backtest tests on the primary cutoff), the model matrix from
    `features_silence` with fixed missing-value flags.
  - `metrics`: ROC AUC, average precision, Brier, log loss, calibration error, reliability bins,
    precision and lift at ten per cent, each with a client-bootstrap interval, and paired
    differences on the same resamples.
  - `churn` (silence): majority and activity-rule baselines, a logistic GLM with odds ratios per
    standard deviation, LightGBM tuned by Optuna inside the earliest cutoff only, isotonic and Platt
    calibration on a held-out fifth of the training clients, SHAP (global and the top reasons per
    client), and the paired ROC AUC gain of the trees over the GLM. The served score is the
    Platt-calibrated one (monotone, so it keeps the trees' ranking), chosen before any evaluation.
  - `survival`: recurrent gaps between active days and time to silence; Kaplan-Meier by channel
    profile and age band with client-bootstrap medians (a vectorised weighted estimator), log-rank
    tests, and a Cox model with client-clustered robust errors (statsmodels `PHReg`) plus a
    Schoenfeld-residual test of proportional hazards.
  - `adoption`: online access and active online use on the aggregate sample, out-of-fold over
    stratified folds (the file has no time axis); GLM odds ratios, LightGBM, gain importance, the
    decile table with lift and cumulative gain; the online counts that define the targets are never
    features (tested).
  - `segments`: Gaussian mixtures by BIC with KMeans silhouettes, bootstrap ARI stability,
    Kruskal-Wallis and chi-square profiles, segment names in words from the two most distinctive
    features, UMAP coordinates for display.
  - `registry` and `cards`: every run logged to an MLflow file store in the zone's workspace, the
    zone's registry and model cards in `<workspace>/models`, `strata promote` publishing
    `models/registry.json` and `docs/model_cards/` from synthetic models only (MLflow run ids left
    out), and `strata ledger check` failing on a card that drifts from the registry.
  - `strata train [all|silence|survival|adoption|segments] [--quick] [--promote]`, `strata promote`,
    `just train` and `just private-train`; CI trains at quick effort.
- Commands run and results:
  - DoD: `uv run strata train all --promote` (synthetic, full effort): four models registered,
    `models/registry.json` and the four model cards written; `strata ledger check` passes (cards
    match the registry); the privacy scan of the published files is clean.
  - DoD: `tests/test_models.py::test_no_feature_reads_past_its_cutoff` rewrites every event after
    the earliest cutoff (drops half, triples the remaining amounts and duplicates them), rebuilds
    the features with dbt and requires that cutoff's features and model matrix to be unchanged,
    while the later cutoffs' features change. Passes.
  - DoD: every metric of the registry and the cards carries a client-bootstrap interval.
  - `uv run strata --private train all` (full effort): exit 0; registry, cards, MLflow runs and
    scores written under `STRATA_PRIVATE_DIR` (output in the private log; only the model names and
    the exit code were read).
  - `just check`: exit 0 (374 passed, 1 skipped, coverage 96.20%).
- Decisions:
  - ADR 0003 gained an implementation section: the backtest protocol, the two-month overlap of
    consecutive outcome windows (stated in every card), the leakage test and the cross-sectional
    validation of the adoption models.
  - lifelines 0.30.3 caps pandas below version three; it is pinned through `pandas>=3.0` in
    `pyproject.toml`, which resolves lifelines 0.30.0. The robust Cox model uses statsmodels,
    because lifelines' clustered sandwich estimator took half a minute on the synthetic gaps.
  - MLflow 3 requires an opt-in to the plain file store that ARCHITECTURE section 10 specifies;
    the registry sets `MLFLOW_ALLOW_FILE_STORE`.
  - Published registries drop MLflow run ids: they only point into a local store, and their
    thirty-two hexadecimal characters would trip the guard's conservative identifier pattern on
    machines without the extract.
- Stubbed or skipped, and why: SHAP is computed for the primary backtest only; no reliability plot
  image is drawn (the bins are in the results file, for the dashboard). Model cards for real data
  stay in the private folder by design.
- Open questions for the architect: none.
- Next: M6.

### M6: Anomaly layer and review queue — done — 2026-10-18

- Built:
  - `strata/models/anomaly/rules.yaml` and `rules.py`: six triage rules (large cash deposit,
    structuring, deposit velocity, pass-through, dormant-then-burst, cash intensity), each hit with
    its evidence; families of rules for pseudo-labels (clients hit by two families or more). The
    thresholds dbt also uses are the dbt project variables, and a test keeps them equal.
  - `scores.py`: an Isolation Forest on `features_anomaly`, an empirical-Bayes model of each
    client's log amounts per category (shrinkage toward the category, robust z), per-client Benford
    tests, a combined score with reasons on every alert, and the alert-budget selector.
  - `run.py` and `strata anomaly run|queue|feedback`: the ranked queue and its evaluation in the
    workspace, the budgeted queue in the review store, precision at k against pseudo-labels (for
    the combined score and for the detectors alone, which never saw the rules) and against analyst
    feedback. The CLI shows client ids for synthetic data only.
  - `strata/review/`: SQLAlchemy models (runs, alerts, feedback, annotations) and Alembic
    migrations; PostgreSQL through `STRATA_PG_URL` in the public zone, a SQLite file otherwise, and
    always a SQLite file under `STRATA_PRIVATE_DIR` in the private zone (ADR 0011).
  - `docs/methods/anomaly.md`. The README, changelog, privacy notes, the `anomaly` recipe and the CI
    step land with the M7 documentation commit, because they share files with it.
- Commands run and results:
  - DoD: `uv run strata anomaly run --budget 50` on synthetic data: a ranked queue of fifty alerts
    stored in the review store, with precision at k against pseudo-labels for the combined score and
    for the detectors alone; on this synthetic extract the queue also holds most of the planted
    structuring and pass-through clients.
  - DoD: feedback round-trips through the database: `strata anomaly feedback <alert> confirmed`,
    then `strata anomaly queue` shows the label and the precision against feedback;
    `tests/test_anomaly.py::test_feedback_round_trips_through_the_store` also migrates a fresh store
    down and up again.
  - DoD: rules are unit-tested on hand-built cases (structuring, pass-through, velocity, large
    deposits, dormancy, cash intensity, pseudo-labels), and a test requires the queue to find at
    least four in five planted scenario clients within its budget.
  - `uv run strata --private anomaly run --budget 50`: exit 0; the queue and the store are under
    `STRATA_PRIVATE_DIR` (checked by script without reading the queue).
  - `just check`: exit 0 (386 passed, 1 skipped, coverage 93.70%).
- Decisions: ADR 0011 (real alerts never go to a database server). The served alert reasons quote
  amounts and dates of the client concerned; that is the evidence a reviewer needs, and it stays in
  the zone of the data.
- Stubbed or skipped, and why: the dashboard's review screens are M8; until then feedback goes
  through the CLI or the API.
- Open questions for the architect: the `cash_intensive` rule fires for many branch-only clients of
  the synthetic data (no card spend at all); should it require some card spend, or keep a separate
  rule for branch-only clients?
- Next: M7.

### M7: API and Compose — done — 2026-10-18

- Built:
  - `strata/api/app.py` (`create_app(settings, zone)`, `strata serve`): `/health` (versions, data
    fingerprint and mapping, model versions), `/clients/{id}/360` (profile, monthly channel
    timeline, segment, silence risk with its top SHAP reasons, open alerts), `/segments`,
    `/risk/silence?top=k`, `/adoption/deciles`, `/alerts?budget=k` with
    `POST /alerts/{id}/feedback`, `/calendar/effects` (both models, each with its interval label),
    `POST /markov/simulate` (share, age limit, target state, horizon; computed from the age-band
    counts of the simulator inputs) and `/ledger/claims` (public claims and verdicts). A read-only
    DuckDB connection per request, Prometheus counters and latency histograms by route template
    (never by raw path, so no client id reaches the metrics), OpenAPI docs, an optional API key.
  - `Dockerfile.api` (the public workspace built into the image at quick effort; the alert queue is
    stored in PostgreSQL at start-up), `Dockerfile.mlflow` (the MLflow UI over the image's runs,
    read-only), `docker-compose.yml` (API, PostgreSQL, MLflow UI, dashboard placeholder; host ports
    default to 8000, 5432, 5000 and 5173 and can be moved with `STRATA_*_PORT`), `.dockerignore`
    (no `.env*`, workspace, data or database file can enter a build context), `docker/smoke.py`
    (every endpoint must answer) and a CI job that builds the stack and runs the smoke test.
  - ADR 0012 (the API lives in the package; the stack runs on synthetic data only).
- Commands run and results:
  - API tests: `tests/test_api.py` (TestClient over httpx) covers every endpoint, the feedback round
    trip, validation errors, missing artefacts, the API key and the metrics' labels; they run in the
    suite, so CI's test job runs them. 9 passed. The review-store migration is also rendered as
    PostgreSQL DDL offline in a test.
  - DoD: `docker compose up --build --detach` (with `STRATA_PG_PORT=15432` and
    `STRATA_DASHBOARD_PORT=15173`, because a local PostgreSQL service and another program already
    listen on 5432 and 5173 on this machine), then `python docker/smoke.py http://127.0.0.1:8000`:
    every endpoint answered (twelve checks); the API and PostgreSQL containers were healthy, the
    MLflow UI and the dashboard placeholder answered, and PostgreSQL held the run, its alerts, the
    smoke test's feedback and the Alembic schema version. The stack was then taken down with its
    volumes; the two built images stay for a quick restart.
  - The Docker engine answered every request with an internal server error for about an hour of
    the session (Docker Desktop is shared with another project); the stack was built once it
    recovered. Nothing in Docker's configuration was changed.
  - `just check`: exit 0 (396 passed, 1 skipped, coverage 97.37%).
- Decisions: ADR 0012. The image bakes the synthetic workspace at build time, so start-up only
  stores the alert queue and starts the server; the MLflow UI needs an empty `mlruns/models`
  folder to open a read-only store, which the image creates.
- Stubbed or skipped, and why: the claims and models inside the image are at quick effort (they say
  so); the dashboard is a placeholder page until M8; CI has not run because nothing is pushed.
- Open questions for the architect: Strata's default host ports 5432 and 5173 are taken on this
  machine (a native PostgreSQL service and a node process), so the stack needs the port overrides
  here; should the defaults move, or stay as specified?
- Next: stop and report (end of this leg).

### M7 review: the architect's answers applied — done — 2026-10-18

- Built:
  - The cash rule is split: `cash_intensive` now requires card spend above zero (deposits at least
    five times card spend and at least the deposit floor), and `cash_only_high_volume` covers
    clients with no card spend and branch deposits at or above the large-deposit threshold; both
    documented in `rules.yaml` and the monitoring method, with hand-built cases and the dbt-variable
    consistency test.
  - The README's run section and `PRIVACY.md` say that machine port overrides (`STRATA_*_PORT`)
    live in `.env.local`, which `just up` passes to Docker Compose; the defaults stay 8000, 5432,
    5000 and 5173.
- Commands run and results: the synthetic queue was re-run (both rules fire, on different
  clients, and the queue's precision against pseudo-labels did not drop); `tests/test_anomaly.py`,
  with the planted-scenario test, passes. The full `just check` of this leg is recorded under M9.
- Next: M8.

### M8: Dashboard — done — 2026-10-19

- Built:
  - `apps/dashboard`: React 19, TypeScript strict, Vite, TanStack Query, ECharts (only the bar,
    line, scatter and heatmap charts and the components they use), Tailwind with design tokens (CSS
    variables with a light and a dark set; the system preference or a choice remembered per
    browser; text tokens at WCAG AA contrast on the surfaces and tints they sit on), Radix sliders,
    KaTeX, pnpm with a lock file, a hash router and lazy-loaded pages.
  - Nine pages: Overview (KPIs, channel mix, calendar heatmap of daily events), Findings (each
    allow-listed claim with its values, intervals, sample size and robustness badge; the
    verdict-only claim as words only: verdict, stratifier names, mechanism), Customer 360 (profile,
    monthly channel timeline, segment, silence risk with SHAP reasons, open alerts), Segments
    (embedding map, profiles, stability), Risk (models compared with client-bootstrap intervals,
    calibration, lift, the ranked list with reasons), Adoption (deciles and cumulative gain), Alerts
    (the budgeted queue with reasons and analyst feedback), Simulator (share and age sliders and the
    target state driving `POST /markov/simulate`, posterior bands for policy and baseline) and
    Methods (KaTeX equations).
  - Data only at run time: every figure, the public claims included, is fetched from the API;
    `scripts/check-bundle.mjs` runs inside `pnpm build` and fails it if a built file holds a value
    or the fingerprint of the public ledger, an identifier-shaped token, a synthetic client id or a
    data file.
  - Tests: Vitest and Testing Library (formatting, themes, claim cards with a check that the
    verdict-only card holds no digit, every page against a mocked API, the feedback call, the
    simulator posting each slider change); Storybook with the accessibility addon and illustrative
    fixtures (badges, KPI tile, claim cards, slider, equation, charts); Playwright smoke tests on a
    desktop and a phone viewport (each page renders from the API with no failure, has no serious or
    critical axe violation in either theme and no sideways scroll; the simulator posts its sliders;
    feedback on an alert is stored).
  - `Dockerfile.dashboard` (the pnpm build served by nginx, `/api/` proxied on the same origin,
    content-security and nosniff headers) replaces the Compose placeholder. The API gained
    `/overview`, `/segments/points` and `/risk/silence/evaluation`, a CORS allowlist for other
    set-ups, and the question and units of each public claim; `docker/smoke.py` covers them.
  - CI: a `dashboard` job (types, tests, build with the bundle check, Storybook) and Playwright in
    the Compose job; `just dashboard`, `just dashboard-check` (part of `just check`) and
    `just dashboard-e2e`.
  - `scripts/screenshots.mjs` refuses to capture anything unless the API reports the public zone
    and a synthetic source; `docs/images/dashboard-*.png` are its captures (the Findings page is
    deliberately not captured).
  - Segment names now take their second word from another channel (no more "no branch use / low
    branch monetary"); the synthetic registry and the segments card were re-published.
  - ADR 0006 expanded with the decisions above.
- Commands run and results:
  - `corepack pnpm build`: pass (type check, Vite build, bundle check: no data in the bundle).
  - `corepack pnpm test`: 5 files, 24 tests passed.
  - `corepack pnpm storybook:build`: pass.
  - DoD, Playwright against Compose: `docker compose --env-file .env.local up --build --detach`
    (with `STRATA_PG_PORT` in `.env.local`, because a local PostgreSQL service holds 5432; every
    other port at its default), `python docker/smoke.py http://127.0.0.1:8000` (every check
    answered), then `STRATA_DASHBOARD_URL=http://127.0.0.1:5173 STRATA_E2E_CHANNEL=msedge pnpm e2e`:
    24 passed (twelve tests on desktop and on phone). The same suite also passed against
    `strata serve` with `vite preview` before the image existed. The stack was then taken down with
    its volumes.
  - DoD: the Findings page renders the claims with intervals and robustness badges and the
    verdict-only claim without a digit (unit test and smoke test); the Simulator page drives
    `/markov/simulate` (unit test and smoke test).
  - `just check`: see the M9 entry (one run covered both milestones).
- Decisions: ADR 0006 (runtime data and the bundle check, one origin through nginx, verdict-only
  claims as words, tokens, weight, tests). Playwright drives the installed Edge locally
  (`STRATA_E2E_CHANNEL=msedge`) instead of downloading a browser; CI installs Chromium.
- Stubbed or skipped, and why: nothing in the dashboard. The Docker engine answered with internal
  server errors for a while again; I waited and did not restart it. CI has not run (nothing pushed).
- Open questions for the architect: pnpm's default content store was created at the root of the
  drive that holds the repository (outside the repository, like the uv cache) when the dashboard's
  packages were first installed, and another project on the machine has used it since, so I left it
  in place; setting `store-dir` in `apps/dashboard/.npmrc` would keep future installs inside the
  repository. Your call.
- Next: M9.

### M9: Orchestration and the private twin — done — 2026-10-19

- Built:
  - `orchestration/airflow/dags/strata_pipeline.py` (the plan as plain data: tasks, their `strata`
    arguments, their upstream tasks) and `strata_daily.py` (Airflow 3, one `BashOperator` per task,
    one task at a time, no retries): `synth` (public zone only), `ingest`, `dq`, `dbt_seed`,
    `dbt_run` and `dbt_test` (all but the feature marts), `features`, `train_silence`,
    `train_survival`, `train_adoption`, `train_segments`, `anomaly`, `evaluate`, `ledger_build`,
    `analyse`, `report`; `STRATA_DAG_ZONE`, `STRATA_DAG_EFFORT`, `STRATA_BIN` and
    `STRATA_DAG_SYNTH_DIR` switch it.
  - `strata evaluate` (`strata/models/evaluate.py`): the gate between the models and the ledger;
    every model trained on the data now in the warehouse (fingerprint and mapping) and better than
    chance (served silence score and adoption models: the lower end of the AUC interval above one
    half; survival concordance above one half; segment stability above zero). It prints checks and
    outcomes, never values.
  - `Dockerfile.airflow` (Airflow 3 in its own environment, pinned by its constraints file, beside
    Strata's environment from `uv.lock`), the Compose service `airflow` under the profile
    `orchestration`, `just dag-test`, and a CI job running `airflow dags test strata_daily`.
  - `strata --private synth-twin` (`strata/synthetic/twin.py`): a differentially private Gaussian
    copula of `marts.client_360` (Laplace noise on histograms over public bins and on Kendall's
    tau-a between every pair of columns, projected to a correlation matrix), sampled at a thousand
    rows, with fidelity metrics (Kolmogorov-Smirnov and total-variation distances, Spearman gaps,
    a real-or-twin classifier's AUC) and privacy metrics (distances to the closest training record
    against a holdout, binned matches, a distance-based membership-inference attack). It refuses CI,
    any destination inside the repository (before anything is read, whatever the zone), any
    destination outside `STRATA_PRIVATE_DIR` and the public zone; the twin carries no synthetic
    marker; the command prints file names only.
  - ADR 0005 expanded (the plan, the tasks, the gate, one task at a time, the image), ADR 0013 (the
    twin, proposed); `PRIVACY.md`, the README and the changelog describe all three.
- Commands run and results:
  - DoD: `docker compose --env-file .env.local --profile orchestration run --rm airflow dags test
    strata_daily` in the built image, on a fresh workspace volume: every one of the sixteen tasks
    succeeded and the DAG run ended `success` (exit 0). The first two attempts failed and were
    fixed: Airflow 3 does not put the DAG folder on the import path (the DAG file now adds its own
    folder), and `dbt run` does not load seeds (hence `dbt_seed`).
  - DoD: `uv run strata --private synth-twin` (through the private-run helper, output in the
    private log): exit 0; `twin.csv` and `metrics.json` are under `<STRATA_PRIVATE_DIR>/twin`.
    `tests/test_twin.py` asserts every refusal (inside the repository in both zones, through the CLI
    too, in CI, outside the private folder, in the public zone), and tests the copula, the sampling
    and the metrics on generated frames: 19 passed.
  - The twin's metrics were read privately as yes-or-no flags only: at every budget tried (from a
    half to sixteen) the membership-inference attack stays at chance and no twin row sits closer to
    a training row than holdout rows do; fidelity is poor at the default budget of one (most
    marginals visibly distorted, correlations mostly noise) and improves with the budget. ADR 0013
    states this in words.
  - `strata evaluate`: public zone, nine checks passed; `strata --private evaluate`: exit 0, nine
    passed.
  - `tests/test_pipeline.py` (the plan, the zones, every task a parsing `strata` command, and the DAG
    file wired against a stand-in for Airflow) and `tests/test_evaluate.py`: 19 and 13 passed.
  - `just check` (one run for M8 and M9): lint and types clean, 450 passed and 1 skipped with
    coverage 97.33%, privacy scan clean, ledger check clean; `dashboard-check` then failed
    because, run from the repository root, `corepack` picked a pnpm of its own instead of the one
    `package.json` pins. The dashboard recipes now run inside `apps/dashboard`, and
    `just dashboard-check` passes; the full `just check` is re-run at the end of M10.
- Decisions: ADR 0005 (an own image rather than the official one, because Airflow's pins and
  Strata's lock disagree; `dbt_seed` as a task; the evaluation gate; one task at a time). ADR 0013
  (the twin reads the client table of the private warehouse rather than the raw files; noise on
  the correlations as well as the marginals, since a copula learns both; bounded neighbouring
  tables with the number of clients public; a thousand rows by default).
- Stubbed or skipped, and why: the DAG has not run in the private zone: Airflow does not run on
  this Windows machine, and running it in a container would mean mounting the extract into Docker,
  which I did not do. The private zone runs through the same CLI steps (`just private-*`).
- Open questions for the architect: accept ADR 0013 (it is marked proposed), including its
  paragraph describing the private sweep in words, or ask me to remove that paragraph; whether the
  twin should move to the Gaussian mechanism for the correlations or fewer columns to buy fidelity
  at the same budget.
- Next: M10.

### M10: Documentation and demo — done — 2026-10-19

- Built:
  - The README rewritten: what Strata is and why it exists, the stages, a Mermaid architecture
    diagram with the two zones, how a source plugs in, how it is checked, screenshots of the
    synthetic demo, the ledger-governed findings section (rendered, unchanged), honesty and
    limitations, privacy, the legacy bugs, the quick start with `just demo`, the recipes and the
    layout.
  - `just demo` (synthetic extract, ingest, quality, dbt, every claim, the analysis modules, the
    models at quick effort and their gate, the monitoring queue, the demo report, then the stack in
    the background and `docker/smoke.py --compose`, which asks Compose where the API and the
    dashboard are published and checks both) and `just down`. The demo neither reads the extract
    nor changes a tracked file (it does not promote).
  - `docs/CV_NOTES.md`, rendered from `docs/findings/_cv_notes.md.j2` by `strata ledger render`:
    every bullet ends with its evidence (a published claim, a key of the public model registry or a
    file of the repository), and `strata ledger check` now fails on its drift, on a hand-typed
    number and on evidence that does not exist (`tests/test_cv_notes.py`). Numbers come only from
    allow-listed claims; the modelling lines carry none, because model metrics on real data are not
    allow-listed.
  - The data card is complete (provenance, composition, collection and coverage, biases and
    limitations, intended and excluded uses, maintenance, and the generated table of assumptions);
    the model cards gained their training data, ethical considerations and how to reproduce and
    refresh them (re-published from the synthetic registry); ADRs 0004 and 0012 point at the twin's
    record and the dashboard image; the changelog covers the leg.
- Commands run and results:
  - DoD: a clean clone (`git clone` of the branch into a scratch folder outside the repository; no
    `.env.local`, no `STRATA_*` variable except `STRATA_PG_PORT=15432`, because a local PostgreSQL
    service holds 5432 on this machine): `just demo`: exit 0 in about fifteen minutes; the
    evaluation gate passed its nine checks and `docker/smoke.py --compose` answered every check
    (fifteen API calls, the dashboard page and the API through the dashboard's proxy); the
    clone's tracked files were unchanged afterwards.
    The Playwright smoke then passed against that stack (24 of 24, desktop and phone), and
    `just down` removed the stack and its volumes; the clone was deleted.
  - DoD: `strata ledger check`: documents match the ledger; no hand-typed numbers.
  - DoD, CI green: not verifiable, because nothing is pushed. Every CI job has been run locally in
    this leg: lint, types and tests (`just check`), the privacy scan, the ledger check, the
    warehouse chain (in the demo and in the DAG test), the dashboard checks, the Compose stack with
    the smoke and Playwright tests (M8) and the DAG test (M9).
  - `just check` (the full recipe, after the last commit of the leg): exit 0. Lint and strict
    types clean; 459 passed and 1 skipped, coverage 97.35%; privacy scan clean; ledger check
    clean; dashboard: 24 unit tests passed, production build with the bundle check (no data in
    the bundle) and the Storybook build. This run is also the green `just check` for M8 and M9.
- Decisions: the CV notes are a governed document like the findings; the dashboard's CV line says
  "customer profiles" instead of the digits of "customer 360", which the hand-typed-number rule
  would refuse in prose; the demo runs at quick effort.
- Stubbed or skipped, and why: CI has not run (nothing pushed). The Findings page is deliberately
  absent from the screenshots (real allow-listed values belong in the ledger, not in images).
- Side effect repaired: running `corepack pnpm --dir apps/dashboard` from the repository root
  (the first form of the dashboard recipes) made corepack fetch the newest pnpm into its user cache
  and record it as its default; that release does not start under this machine's corepack, which
  would have broken any unpinned `corepack pnpm` call on the machine. I removed that cache entry
  and the default it recorded (both created by this run, by their timestamps); the recipes now run
  inside `apps/dashboard`, where the pinned pnpm is used.
- Open questions for the architect: should some real-data model metrics be allow-listed (for
  example the served silence score's Brier score and lift with their intervals, and the segments'
  stability), so that the modelling CV line can carry numbers? ADR 0013 is proposed and needs your
  decision (see M9).
- Next: stop and report (end of the rebuild's milestones).

### M11: Polish after the final review — done — 2026-10-19

- Built:
  - The architect's answers: ADR 0013 accepted (with its sweep paragraph, now revised), the Airflow
    image deviation of ADR 0005 accepted, and one README line on where pnpm keeps its store (no
    `store-dir` inside the repository).
  - A third publication mode, `metrics_only`, for the quality of the models trained on the real
    extract: `silence_model_quality` (the served silence score's Brier score, calibration error and
    lift in the top tenth on a later cutoff) and `segment_stability` (the segments' bootstrap
    adjusted Rand index), each with its interval. The claims read what `strata train` wrote for the
    zone, and only from a model trained on the same data and mapping. Enforced in the allowlist
    (`METRICS_ONLY_KEYS`, `METRICS_ONLY_META`, `_metrics_violations`), the build (`metrics_view`
    keeps the allow-listed metrics, their intervals and sizes and how they were measured, refuses a
    quick model or a sample under the floor), the renderer (only `ledger`, `ci` and `raw` for those
    metrics; no table, no robustness placeholder), the ledger check and the commit guard (both
    through `ledger_violations`). Published from the private zone with
    `strata --private ledger build --public --only silence_model_quality --only segment_stability`;
    the earlier public claims and the verdict were unchanged. The CV notes' modelling lines now carry
    those numbers. Because the claims read trained models, the API image, CI and `just demo` now
    train before they build the ledger (the DAG already did).
  - The twin, time-boxed: the analytic Gaussian mechanism as an option (`--mechanism gaussian`,
    `--delta`) and a reduced column set (`--columns core`). In a private experiment over five
    seeds at the same budgets, judged on the reduced set's columns, the reduced Laplace twin kept
    clearly more of the structure than the full one at a budget of one, a real-or-twin classifier
    no longer separated it from the data, and it never raised a privacy flag; the Gaussian mechanism
    helped the full set, added little on the reduced one and once let the membership-inference
    attack rise above chance. The reduced Laplace twin is the new default; ADR 0013 records the
    experiment in words. The private twin was rebuilt with the new default (exit 0; membership
    inference at chance).
  - Cheap stubs closed: the silence model computes SHAP for every backtest (its card lists the three
    largest drivers of each) and draws a reliability diagram of the served score and the GLM
    (`strata.models.plots`), which `strata promote` copies beside the public card. The synthetic
    silence model was retrained and the cards re-published; the private silence model was retrained
    too (its image and per-backtest SHAP stay in the private folder), and rebuilding the published
    metrics from it gave identical values, so the public ledger was left as it was.
- Commands run and results:
  - `tests/test_metrics_only.py`: 9 passed (allowlist, view, rules, guard, renderer, publication,
    stale and missing models); the synthetic ledger fixture now trains the silence and segment
    models first, so every registered claim still computes on synthetic data.
  - `tests/test_twin.py`: 24 passed (the refusals, the Gaussian calibration against the classical
    bound, the new default); `tests/test_models.py`: 17 passed (per-backtest SHAP, the PNG).
  - `strata ledger check`: clean, with the CV notes rendered from the new claims.
  - `just check` (the full recipe, after the last code commit): exit 0. Lint and strict types clean;
    475 passed and 1 skipped, coverage 97.31%; privacy scan clean; ledger check clean; dashboard
    checks passed (24 unit tests, the production build with its bundle check, Storybook).
  - Final clean-clone demo (`git clone` of the branch into a scratch folder, no `.env.local`, only
    `STRATA_PG_PORT=15432` set): every host stage passed (synthetic extract, ingest, quality, dbt,
    training, the gate with its nine checks, the ledger with both model-quality claims computed,
    the analysis modules, the queue and the report); `docker compose up` then failed because
    the shared Docker engine answered every request with an internal server error, and it was
    still doing so hours later. I did not restart it. In its place, the parts of the stack this
    leg changed were verified outside Docker: `strata serve` with the dashboard's production
    build under `vite preview` (the API proxied as in nginx) served the public ledger with the
    two metrics-only claims, `docker/smoke.py` answered every check including the dashboard
    and its proxy, and the Playwright smoke passed 24 of 24 (desktop and phone, axe in both
    themes). The last complete clean-clone `just demo`, Compose included, is the M10 one; the
    image's only change since is that it trains before it builds the ledger, the order the
    demo's host stages ran successfully here.
- Decisions: ADR 0002 gains the `metrics_only` mode; ADR 0013 revised (Gaussian option, reduced
  default). The model-quality claims carry an interval label of their own for the segments
  (the range of refits, not a confidence interval of the mean).
- Stubbed or skipped, and why: none of the listed polish items; CI still has not run (nothing is
  pushed).
- Open questions for the architect: none.
- Next: none; this was the last leg.

### Design pass: the dashboard redesigned to the brief — done — 2026-10-19

- Built:
  - `apps/dashboard/src/styles/tokens.css`: every colour, type size, space, radius, material,
    elevation and duration as a token, with light and dark values (system preference, chosen
    theme, or a stamped subtree), nine greys, one accent (AA on both themes), success, warning,
    danger, three verdict colours (holds green, weakens amber, reverses violet), five categorical
    hues at equal OKLCH lightness whose order was chosen with the dataviz palette validator,
    sequential and diverging ends, the system type stack with size-specific tracking and tabular
    figures, the 4px unit, three radii and the translucent material, with
    `prefers-reduced-transparency` and `prefers-contrast: more` honoured. Tailwind's own palette,
    radii, shadows and type scale are cleared, so utilities can only name tokens.
    `src/styles/tokens.test.ts` checks that the two dark blocks agree, that every token the chart
    theme reads exists, AA contrast for every text pair, and the palette's equal lightness and
    colour-vision separation.
  - The ECharts theme generated from the tokens on each chart's own element (`src/charts/theme.ts`),
    merged into the option so one chart instance lives for its page: updates go through ECharts'
    transitions with series matched by id, the entrance plays on a page's first paint only,
    reduced motion turns animation off, options name colours by token. Interval whiskers and
    reference lines as custom series (`src/charts/marks.ts`).
  - Motion and interaction: Motion springs (critically damped by default, bounce only after a drag
    release, a cross-fade under reduced motion); buttons and links press on pointer-down; a
    segmented control whose selection slides and can be redirected; the evidence drawer (Radix
    Dialog, slides from the right and returns the same way, the top bar turns solid while it is
    open, focus returns to the row's button); the simulator's slider (one-to-one tracking under
    pointer capture with the grab offset kept, rubber-banding past the ends, the release velocity
    handed to a spring, arrows by one step and Shift by ten), with bands that follow it every frame
    by interpolating between policies the API has computed (`src/simulator.ts`) while the network
    call is debounced.
  - The shell: a sidebar on wide screens and a scrolling page row on narrow ones, a translucent top
    bar that names the page and its data ("Synthetic demo", plus "Claims: reference extract" on
    Findings) with a soft edge once content scrolls under it, and Lucide icons only.
  - Every page to the brief's structure: Overview (five numbers with their intervals or samples in a
    left column; the channel mix with interval whiskers and the calendar of daily activity on the
    right); Findings (titled "Findings (n claims)", each card titled with its question, values with
    intervals in one format, the verdict and its explanation, n, events and period in a caption,
    the verdict-only card digit-free); Customer profile (search, section anchors, profile,
    timeline, segment, silence risk with its three reasons, open alerts); Segments (the map in the
    neutral ramp with the chosen segment in the accent, the profile table beside it, pointing at
    either chooses the segment, the stability in the table's header); Silence risk (the ranked
    table first, calibration and lift with intervals, the models compared); Adoption (decile tables
    with lift, one cumulative-gain chart: the old second y-axis is gone); Alerts (titled "Alerts (n
    ranked)", evidence and feedback in the drawer); Simulator (sliders left, the policy in words
    above the bands, the outcome at the horizon with its intervals); Methods (prose and equations
    at the reading measure, flush left). Errors say what happened and the command that fixes it,
    loading states are skeletons, empty states name the command that fills them.
  - The API serves the context those pages need, additively: each public claim's method, caveats,
    robustness explanation, event count, period and source; the overview's period and channel-mix
    intervals (only from a ledger built on the data in the warehouse); the segments' embedding and
    client count; the scored population's size; and the description of every interval it serves,
    defined once beside the percentiles that make it (`strata/models/intervals.py`,
    `markov.BAND_INTERVAL`), so dashboard copy never types a coverage level.
  - Storybook shows every story in both themes side by side (foundations, controls, feedback,
    layout, findings, charts, drawer). `scripts/screenshots.mjs` captures every page but Findings
    in both themes at desktop and phone width once every chart reports it has finished drawing;
    its captures replace `docs/images/`, and the README shows four of them.
  - `docs/DESIGN_NOTES.md` lists every departure from the brief and why; ADR 0006 records the stack
    additions (tokens, the merged chart theme, Motion, Lucide, the own slider, Radix Dialog) and the
    alternatives set aside; the changelog covers the pass.
- Commands run and results:
  - `uv run pytest tests/test_api.py tests/test_metrics_only.py tests/test_markov.py`: 38 passed
    (the new fields, the stale-ledger guard, the served interval descriptions).
  - `corepack pnpm test`: 9 files, 50 tests passed (tokens, chart theme, formatting, slider
    arithmetic and keys, band interpolation, claim cards, every page against the fixture API, the
    drawer's feedback, the simulator's requests, the theme switch).
  - `corepack pnpm build`: pass; `check-bundle: 78 files, no data in the bundle`.
  - `corepack pnpm storybook:build`: pass.
  - The Docker engine still answers every request with an internal server error (`docker info`),
    so, as in M11, the stack ran outside Docker: `strata serve` on the public synthetic workspace and
    the production build under `vite preview --host 127.0.0.1` (the API proxied as in nginx). Against
    it, `STRATA_E2E_CHANNEL=msedge corepack pnpm e2e`: 26 passed (thirteen tests on desktop and on
    phone: every page renders with no failure, no serious or critical axe violation in either theme
    and no sideways scroll; the claims and the digit-free verdict; the first simulation is the
    policy on screen and a key step posts the next; a pointer drag moves the value and holds it at
    the end; feedback in the drawer is stored and confirmed in place, axe clean with the drawer
    open, focus back on its button after Escape).
  - `node scripts/screenshots.mjs` against the same stack (it checked the public zone and the
    synthetic source first): all captures written, each under the large-file limit.
  - `strata ledger check`: documents match the ledger; no hand-typed numbers.
  - `just check` (the full recipe, after the last code commit): exit 0. Lint and strict types
    clean; 476 passed and 1 skipped, coverage 97.30%; privacy scan clean; ledger check clean;
    dashboard checks passed (50 unit tests, the production build with its bundle check,
    Storybook).
- Decisions (ADR 0006 and `docs/DESIGN_NOTES.md`): Findings stays out of the pictures; the top bar
  adds the claims' source on Findings; verdicts are green, amber and violet; the segment map is
  focus and context; the chart theme is merged, not registered; section headings take a looser
  leading than titles; equations are flush left; the slider thumb and the selected segment carry
  the smallest elevation; a release settles on the nearest step without momentum projection;
  feedback is given in the drawer; the top bar turns solid under a sheet; theme changes are
  instant. New dependencies are locked at releases at least two weeks old (Motion 13.5.1, Lucide
  1.49.0, Radix Dialog 1.1.23); the Radix slider, tabs and tooltip were removed.
- Stubbed or skipped, and why: the Compose stack was not run (Docker engine down; not restarted);
  CI has not run (nothing pushed). `prefers-reduced-transparency` is honoured where the browser
  supports the query; elsewhere the translucent bar stays translucent. Diverging tokens exist but no
  current chart needs them (the silence reasons are all positive contributions).
- Open questions for the architect: whether the Findings page should be pictured after all (with
  the drift risk the notes describe); whether violet reads right for "reverses"; whether the
  interpolated bands while dragging are acceptable as described, or should be marked visually as
  provisional.
- Next: none; stop and report.
