"""The daily pipeline as plain data: the tasks of the Airflow DAG ``strata_daily`` and their order.

Kept free of Airflow and of the ``strata`` package, so that the DAG file can import it inside
Airflow's own environment and the tests can import it without Airflow. Every task is one call of
the ``strata`` command line; the zone and the effort are the only switches.

    synth (public zone only) -> ingest -> dq -> dbt_seed -> dbt_run -> dbt_test -> features
        -> train_silence, train_survival, train_adoption, train_segments, anomaly
        -> evaluate -> ledger_build -> analyse -> report
"""

from __future__ import annotations

import shlex
from dataclasses import dataclass

FEATURES = "features_*"
"""The feature marts, built and tested after the rest of the warehouse has passed its tests."""

MODELS = ("silence", "survival", "adoption", "segments")
ALERT_BUDGET = "50"
ZONES = ("public", "private")


@dataclass(frozen=True)
class Step:
    """One task: its id, the ``strata`` arguments it runs, the tasks it waits for, what it does."""

    task_id: str
    arguments: tuple[str, ...]
    upstream: tuple[str, ...]
    doc: str


def steps(zone: str, *, quick: bool, synthetic_dir: str) -> list[Step]:
    """The tasks of one zone, in an order where every task comes after its upstream tasks."""
    if zone not in ZONES:
        msg = f"zone must be one of {ZONES}"
        raise ValueError(msg)
    effort = ("--quick",) if quick else ()
    out: list[Step] = []
    if zone == "public":
        out.append(
            Step("synth", ("synth", "--out", synthetic_dir), (), "Write the synthetic extract.")
        )
        out.append(
            Step(
                "ingest",
                ("ingest", "--data-dir", synthetic_dir),
                ("synth",),
                "Contracts, raw layer, manifest and reconciliation on the synthetic extract.",
            )
        )
    else:
        out.append(
            Step(
                "ingest",
                ("ingest",),
                (),
                "Contracts, raw layer, manifest and reconciliation on STRATA_DATA_DIR.",
            )
        )
    out += [
        Step("dq", ("dq",), ("ingest",), "Data-quality report; fails the run on errors."),
        Step("dbt_seed", ("dbt", "seed"), ("dq",), "The code maps, bands and calendars."),
        Step(
            "dbt_run",
            ("dbt", "run", "--exclude", FEATURES),
            ("dbt_seed",),
            "Staging, intermediate models and the marts other than the features.",
        ),
        Step(
            "dbt_test",
            ("dbt", "test", "--exclude", FEATURES),
            ("dbt_run",),
            "Every dbt test of the models built so far.",
        ),
        Step(
            "features",
            ("dbt", "build", "--select", FEATURES),
            ("dbt_test",),
            "The feature marts and their tests, including no feature past its cutoff.",
        ),
    ]
    out += [
        Step(
            f"train_{name}",
            ("train", name, *effort),
            ("features",),
            f"Fit and log the {name} model.",
        )
        for name in MODELS
    ]
    out.append(
        Step(
            "anomaly",
            ("anomaly", "run", "--budget", ALERT_BUDGET),
            ("features",),
            "The monitoring queue for review, stored in the review store.",
        )
    )
    out += [
        Step(
            "evaluate",
            ("evaluate",),
            (*(f"train_{name}" for name in MODELS), "anomaly"),
            "Gate: every model trained on this data and better than chance.",
        ),
        Step(
            "ledger_build", ("ledger", "build", *effort), ("evaluate",), "Every registered claim."
        ),
        Step(
            "analyse",
            ("analyse", "all", *effort),
            ("ledger_build",),
            "Analysis modules: their claims, figures and the simulator inputs.",
        ),
        Step(
            "report",
            ("report", "--variant", zone),
            ("analyse",),
            "The report of the zone: public demo, or every claim in the private folder.",
        ),
    ]
    return out


def command(step: Step, zone: str, executable: str = "strata") -> str:
    """The shell command of one task; the private zone adds ``--private`` to every call."""
    zone_flag = ["--private"] if zone == "private" else []
    return shlex.join([executable, *zone_flag, *step.arguments])
