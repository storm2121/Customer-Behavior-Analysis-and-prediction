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
