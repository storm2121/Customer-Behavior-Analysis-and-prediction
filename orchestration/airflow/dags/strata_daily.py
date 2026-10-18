"""strata_daily: the whole Strata pipeline once a day, every task a call of the strata CLI.

Environment (all optional):

- ``STRATA_DAG_ZONE``: ``public`` (synthetic data, the default) or ``private`` (the real extract
  through ``STRATA_DATA_DIR``; outputs only under ``STRATA_PRIVATE_DIR``).
- ``STRATA_DAG_EFFORT``: ``full`` (default) or ``quick`` (Compose and tests).
- ``STRATA_BIN``: the strata executable (Airflow and Strata live in separate environments).
- ``STRATA_DAG_SYNTH_DIR``: where the public run writes its synthetic extract.

The tasks run one at a time: DuckDB takes one writer, and the registry is updated in place.
"""

from __future__ import annotations

import os
import sys
from datetime import UTC, datetime
from pathlib import Path

from airflow.providers.standard.operators.bash import BashOperator
from airflow.sdk import DAG

# Airflow 3 does not put the DAG folder on the import path; the plan lives beside this file.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from strata_pipeline import command, steps

ZONE = os.environ.get("STRATA_DAG_ZONE", "public")
QUICK = os.environ.get("STRATA_DAG_EFFORT", "full") == "quick"
STRATA = os.environ.get("STRATA_BIN", "strata")
SYNTH_DIR = os.environ.get(
    "STRATA_DAG_SYNTH_DIR", str(Path(os.environ.get("STRATA_WORK_DIR", ".strata")) / "synth")
)

with DAG(
    dag_id="strata_daily",
    description="ingest, quality, dbt, features, models, gate, ledger, analysis and report",
    schedule="@daily",
    start_date=datetime(2026, 1, 1, tzinfo=UTC),
    catchup=False,
    max_active_runs=1,
    max_active_tasks=1,
    default_args={"retries": 0},
    tags=["strata", ZONE],
    doc_md=__doc__,
) as dag:
    plan = steps(ZONE, quick=QUICK, synthetic_dir=SYNTH_DIR)
    tasks = {
        step.task_id: BashOperator(
            task_id=step.task_id,
            bash_command=command(step, ZONE, STRATA),
            doc_md=step.doc,
        )
        for step in plan
    }
    for step in plan:
        for upstream in step.upstream:
            tasks[upstream] >> tasks[step.task_id]
