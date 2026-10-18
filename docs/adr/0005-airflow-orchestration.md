# ADR 0005: Airflow for orchestration, running only in Docker Compose

Status: accepted (architect, 2026-10-18). Filled in M0; expanded in M9, when the DAG was built.

## Context

The pipeline has a clear DAG (ingest, quality, dbt, features, training, evaluation, ledger, report).
Airflow is the orchestrator most job descriptions name; it does not run natively on Windows.

## Decision

A single DAG `strata_daily` whose tasks call the `strata` CLI, executed in Compose with the official
image and tested with `airflow dags test`. Local development runs the same steps through `just`
recipes without Airflow.

Because every task is a CLI call, the DAG holds no business logic: what Airflow runs is exactly what
a developer runs, and the DAG can be replaced without touching the pipeline.

### How it was built (M9)

- **The plan is data.** `orchestration/airflow/dags/strata_pipeline.py` lists the tasks, their
  `strata` arguments and their upstream tasks, without importing Airflow or the package;
  `strata_daily.py` turns each into a `BashOperator`. Tests check the plan's order, the zone and
  effort switches, that every task parses as a `strata` command, and, with a stand-in for the two
  Airflow names, that the DAG file wires exactly that plan.
- **Tasks.** `synth` (public zone only), `ingest`, `dq`, `dbt_seed`, `dbt_run` and `dbt_test` (every
  model but the feature marts), `features` (the feature marts and their tests, including no feature
  past its cutoff), one `train_*` task per model and `anomaly`, then `evaluate`, `ledger_build`,
  `analyse` and `report`. `dbt_seed` is a task of its own because `dbt run` does not load seeds.
- **An evaluation gate.** `strata evaluate` fails the run before the ledger and the report unless
  every model was trained on the data now in the warehouse (fingerprint and mapping) and clears a
  floor any working model clears: the served silence score and the adoption models rank better
  than chance (the lower end of their AUC intervals above one half), the survival model's
  concordance is above one half, the segments are stable under bootstrap refits. Its output names
  checks, never values, so it can run in the private zone.
- **One task at a time.** DuckDB takes a single writer and the registry is updated in place, so the
  DAG sets `max_active_tasks` to one; the graph still shows the model tasks in parallel. No retries:
  a failed day should be looked at, not repeated.
- **Switches.** `STRATA_DAG_ZONE` (public by default; private runs add `--private` to every call),
  `STRATA_DAG_EFFORT` (quick in Compose), `STRATA_BIN` and `STRATA_DAG_SYNTH_DIR`.
- **The image (a deviation from "the official image").** `Dockerfile.airflow` starts from the same
  Python image as the API and installs Airflow 3 in its own environment, pinned by Airflow's
  constraints file for that Python version, beside Strata's environment from `uv.lock`. Airflow's
  pins and Strata's lock disagree (SQLAlchemy among others), so Strata could not live inside the
  official image's environment anyway; two environments in one image keep both exact. The service
  runs only under the Compose profile `orchestration`; `just dag-test` runs
  `airflow dags test strata_daily`, and the UI (open without a login, synthetic data only) is on
  `STRATA_AIRFLOW_PORT`.
- **Private runs.** Airflow does not run on this Windows machine; the private zone is reached through
  the same plan with `STRATA_DAG_ZONE=private` where Airflow runs next to the extract, or step by step
  with the `just private-*` recipes.

## Alternatives considered

- **Dagster.** Asset-oriented and lighter to run; a better fit technically, less recognisable.
- **Prefect.** Pleasant Python API, smaller footprint, also less recognisable.
- **`just` or cron only.** Sufficient for one machine, but shows nothing about scheduling, retries
  or observability.

## Consequences

Heavier Compose stack, but one honest orchestration story. Dagster or Prefect would be lighter; the
choice is driven by recognisability, recorded here so it can be revisited.
