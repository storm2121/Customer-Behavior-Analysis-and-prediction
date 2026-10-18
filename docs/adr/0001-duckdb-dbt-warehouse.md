# ADR 0001: DuckDB with dbt as the analytical core

Status: accepted (architect, 2026-10-18). Context and consequences expanded in M0 (executing agent).

## Context

The 2024 script merged five CSVs in pandas with per-client aggregates computed inline, which hid a
join that matched zero rows and mixed feature logic with plotting. The data is small (tens of
thousands of rows) but the logic is relational: unions across channels, window functions for recency
and rolling sums, monthly states, time-cut features.

Forces at play:

- **Reviewability.** Feature definitions are the part of the system most likely to be wrong and the
  part reviewers most need to read. Inline pandas chains are hard to review and impossible to test
  one step at a time.
- **One definition per feature.** The same per-client aggregates feed the ledger claims, the risk
  models, the segments and the API. Recomputing them in each consumer invites drift.
- **Size.** The whole extract fits in memory many times over, so a database server would be
  operational weight with no analytical benefit.
- **Platform.** Development happens on Windows; CI and the containers run Linux. Everything in the
  core must install from wheels on both.

## Decision

Load the extracts into a DuckDB file and express every transformation as a dbt model with tests.
Python reads marts; it does not rebuild them. Python 3.12 managed by uv for compatibility with
dbt-duckdb, PyMC and the rest of the stack.

The raw layer is written by the Python ingest (which validates contracts and records a manifest);
everything after it (staging, intermediate, marts) belongs to dbt. Code maps live in dbt seeds so a
reader can see, and review, every assumed code meaning.

## Alternatives considered

- **pandas only, better organised.** Rejected: it keeps logic and orchestration in one place and has
  no native notion of a tested, documented model graph.
- **PostgreSQL as the warehouse.** Rejected for analytics: a server for a file-sized dataset.
  PostgreSQL is still used for operational state (alerts, feedback), where concurrent writes matter.
- **SQLite.** Rejected: weaker analytical SQL, no columnar execution, awkward date handling.
- **Polars.** A good engine, but it offers no equivalent of dbt's tests, documentation and lineage.
  It remains available inside Python code where a dataframe is the right tool.

## Consequences

- SQL becomes reviewable and tested; feature definitions exist once; dbt docs describe the lineage.
- Two toolchains (dbt and Python) and a build step before analysis.
- DuckDB allows one writer at a time: the pipeline writes, while the API opens the file read-only.
- dbt-core and dbt-duckdb versions move together and are locked in `uv.lock`.
- Python 3.12 is the newest interpreter for which the whole stack (dbt, PyMC, LightGBM, Airflow
  client libraries) is expected to ship wheels on Windows; moving to a newer Python is a deliberate
  later step.
