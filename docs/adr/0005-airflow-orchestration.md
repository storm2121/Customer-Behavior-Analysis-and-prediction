# ADR 0005: Airflow for orchestration, running only in Docker Compose

Status: accepted (architect, 2026-10-18). Filled in M0; expanded when the DAG is built in M9.

## Context

The pipeline has a clear DAG (ingest, quality, dbt, features, training, evaluation, ledger, report).
Airflow is the orchestrator most job descriptions name; it does not run natively on Windows.

## Decision

A single DAG `strata_daily` whose tasks call the `strata` CLI, executed in Compose with the official
image and tested with `airflow dags test`. Local development runs the same steps through `just`
recipes without Airflow.

Because every task is a CLI call, the DAG holds no business logic: what Airflow runs is exactly what
a developer runs, and the DAG can be replaced without touching the pipeline.

## Alternatives considered

- **Dagster.** Asset-oriented and lighter to run; a better fit technically, less recognisable.
- **Prefect.** Pleasant Python API, smaller footprint, also less recognisable.
- **`just` or cron only.** Sufficient for one machine, but shows nothing about scheduling, retries
  or observability.

## Consequences

Heavier Compose stack, but one honest orchestration story. Dagster or Prefect would be lighter; the
choice is driven by recognisability, recorded here so it can be revisited.
