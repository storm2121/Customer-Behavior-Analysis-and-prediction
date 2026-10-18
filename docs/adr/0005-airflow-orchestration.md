# ADR 0005: Airflow for orchestration, running only in Docker Compose

Status: accepted (architect, 2026-10-18). Expand during M9.

## Context

The pipeline has a clear DAG (ingest, quality, dbt, features, training, evaluation, ledger, report).
Airflow is the orchestrator most job descriptions name; it does not run natively on Windows.

## Decision

A single DAG `kashf_daily` whose tasks call the `kashf` CLI, executed in Compose with the official
image and tested with `airflow dags test`. Local development runs the same steps through `just`
recipes without Airflow.

## Consequences

Heavier Compose stack, but one honest orchestration story. Dagster or Prefect would be lighter; the
choice is driven by recognisability, recorded here so it can be revisited.
