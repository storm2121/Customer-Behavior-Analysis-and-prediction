# ADR 0001: DuckDB with dbt as the analytical core

Status: accepted (architect, 2026-10-18). Expand the context and consequences during M0.

## Context

The 2024 script merged five CSVs in pandas with per-client aggregates computed inline, which hid a
join that matched zero rows and mixed feature logic with plotting. The data is small (tens of
thousands of rows) but the logic is relational: unions across channels, window functions for recency
and rolling sums, monthly states, time-cut features.

## Decision

Load the extracts into a DuckDB file and express every transformation as a dbt model with tests.
Python reads marts; it does not rebuild them. Python 3.12 managed by uv for compatibility with
dbt-duckdb, PyMC and the rest of the stack.

## Consequences

SQL becomes reviewable and tested; feature definitions exist once; dbt docs describe the lineage.
Cost: two toolchains (dbt and Python) and a build step before analysis.
