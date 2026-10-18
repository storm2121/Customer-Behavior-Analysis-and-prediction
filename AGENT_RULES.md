# Agent rules for the Kashf rebuild

These rules bind every agent session working in this repository. Read `ARCHITECTURE.md` and
`MILESTONES.md` first. When a rule and convenience conflict, the rule wins. When a rule and the
architecture conflict, write an ADR and stop for review.

## 1. Privacy (non-negotiable)

1. The real extract is read only through `kashf.config.Settings.data_dir`, set by `KASHF_DATA_DIR`
   (on this machine: `D:\Data_CIh`). Never hard-code that path in code, tests, docs or notebooks.
2. Never print, log, assert on, or paste an identifier or a raw row from the real data, not in
   terminal output, not in PROGRESS.md, not in commit messages, not in test fixtures.
3. Nothing derived from real rows is written inside the repository, except claims whose ids are in
   `kashf/ledger/allowlist.py`, written to `docs/ledger/claims.public.json` by `kashf ledger build --public`.
   Everything else goes under `KASHF_PRIVATE_DIR` (default `D:\Data_CIh\private_out`).
4. Models trained on real data are private artefacts. Models in the repo are trained on synthetic data.
5. Before every commit run `git status` and `uv run kashf privacy-check --staged`; the pre-commit
   hook must be installed in M0 and never bypassed (`--no-verify` is forbidden).
6. Plots are allowed in the repo only when generated from synthetic data or from allow-listed claims.
7. If in doubt whether something is private, it is private.

## 2. Honesty

1. No number is typed by hand in any document, README section, dashboard copy or report. Numbers
   come from the ledger through placeholders. Hand-typed numbers are a CI failure.
2. Every claim carries n, a confidence interval, its method and its caveats. Every headline claim
   carries a robustness verdict.
3. Context travels with every number: "1,000 anonymised clients, one rolling year, time-based
   validation, synthetic demo". Never present a synthetic-data metric as a real-data result.
4. No causal language. "Associated with", not "drives", unless the method supports it.
5. If a model is not trained, a test is skipped, or a feature is stubbed, PROGRESS.md says so in
   plain words. Never fake a result, a metric, a screenshot or a passing test.
6. The legacy code's bugs are stated openly in `legacy/README.md` and in `CHANGELOG.md`.

## 3. Engineering standards

1. Python 3.12 managed by `uv` (`uv python install 3.12`; `.python-version`; `uv sync`). All
   commands run as `uv run …`. Dependencies are declared in `pyproject.toml` with lower bounds and
   locked in `uv.lock`.
2. Typed code: `mypy --strict` on `kashf/`, `ruff` format and lint clean. Pydantic v2 models at
   boundaries; `pydantic-settings` for configuration; `typer` for the CLI; `structlog` for logging.
3. Tests first for pure functions. `pytest` + `hypothesis`. Everything in `tests/` runs on synthetic
   data generated in a session fixture; no test may require `KASHF_DATA_DIR`. Coverage gate 85% on
   `kashf/`. A marked `@pytest.mark.private` suite may run locally against real data but is skipped
   by default and never asserts on values.
4. Deterministic seeds everywhere. Any randomness takes a `seed` argument.
5. No notebooks as source of truth. Notebooks may live in `notebooks/` for exploration only, with
   outputs stripped by `nbstripout`.
6. SQL lives in dbt models, not in f-strings. Feature logic that must be shared between dbt and
   Python is defined once and tested once.
7. Dashboard: TypeScript strict, ESLint + Prettier, components in Storybook, design tokens, no magic
   numbers in styles, keyboard accessible, axe clean, works at phone width.
8. Docstrings explain why, not what. Comments record decisions and trade-offs, not narration.
9. Keep `legacy/` frozen. Do not refactor it; document it.

## 4. Git

1. Work on branch `feat/kashf-rebuild`. Small conventional commits (`feat:`, `fix:`, `docs:`,
   `test:`, `chore:`, `refactor:`), one concern each.
2. Never push. Never open a PR. Mohamed decides when anything leaves the machine.
3. Never add co-author or "generated with" lines to commits.
4. Never commit `data/`, `*.csv` (except `dbt/seeds/*.csv` and fixtures under 200 rows),
   `.kashf/`, `mlruns/`, `*.duckdb`, `claims.private.json`, or anything under `KASHF_PRIVATE_DIR`.

## 5. Workflow

1. Milestones are done in order (`MILESTONES.md`). A milestone is done only when its definition of
   done passes and `just check` is green.
2. After each milestone, append to `PROGRESS.md`: what was built, exact commands run and their
   results (pass/fail counts, no private values), decisions taken, open questions, what is stubbed.
3. If blocked, record the blocker in `PROGRESS.md` with what was tried, then continue with the next
   item that does not depend on it. Do not spend more than thirty minutes on any single environment
   or installation problem; fall back as the architecture specifies and write the ADR.
4. Deviating from `ARCHITECTURE.md` requires an ADR in `docs/adr/` (`NNNN-title.md`: status,
   context, decision, consequences) before the code.
5. Stop and return a report when M4 is complete, or earlier if context is running low. The architect
   session continues from `PROGRESS.md`.
6. Never modify anything outside this repository except files under `KASHF_PRIVATE_DIR` and the
   project's own virtual environment. Never touch `D:\CV's`.

## 6. Definition of done for the whole rebuild

`just demo` on a clean clone with no real data: generates synthetic data, builds the warehouse, runs
dbt tests, builds and checks the ledger, trains and evaluates every model, renders the public report,
and brings up API, dashboard and Airflow in Compose, with CI green. Separately, with
`KASHF_DATA_DIR` set, `just private-report` produces the full private report and the private twin
with fidelity and privacy metrics, writing nothing into the repository.
