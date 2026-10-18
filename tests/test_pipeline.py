"""The daily DAG: its plan (order, zones, effort) and the DAG file built from it.

Airflow does not run on Windows and is not a dependency of the package, so the DAG file is
imported here against a minimal stand-in for the two Airflow names it uses; the real DAG is
exercised with `airflow dags test` in the Compose image (`just dag-test`).
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from typing import Any, ClassVar

import pytest
from typer.testing import CliRunner

from strata import cli

DAGS = Path(__file__).resolve().parents[1] / "orchestration" / "airflow" / "dags"


def _load(name: str) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(name, DAGS / f"{name}.py")
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


pipeline = _load("strata_pipeline")


def _ids(plan: list[Any]) -> list[str]:
    return [step.task_id for step in plan]


def test_the_public_plan_follows_the_architecture() -> None:
    plan = pipeline.steps("public", quick=False, synthetic_dir="synth")
    ids = _ids(plan)
    assert ids[:7] == ["synth", "ingest", "dq", "dbt_seed", "dbt_run", "dbt_test", "features"]
    assert ids[-4:] == ["evaluate", "ledger_build", "analyse", "report"]
    assert {
        "train_silence",
        "train_survival",
        "train_adoption",
        "train_segments",
        "anomaly",
    } <= set(ids)
    seen: set[str] = set()
    for step in plan:
        assert set(step.upstream) <= seen, step.task_id
        seen.add(step.task_id)


def test_the_private_plan_reads_the_extract_and_flags_every_call() -> None:
    plan = pipeline.steps("private", quick=False, synthetic_dir="unused")
    assert "synth" not in _ids(plan)
    commands = [pipeline.command(step, "private") for step in plan]
    assert all(line.startswith("strata --private ") for line in commands)
    assert commands[-1].endswith("report --variant private")
    assert not any("unused" in line for line in commands)


def test_quick_effort_reaches_the_models_the_claims_and_the_analysis() -> None:
    plan = {step.task_id: step for step in pipeline.steps("public", quick=True, synthetic_dir="s")}
    for task in ("train_silence", "train_survival", "train_adoption", "train_segments"):
        assert "--quick" in plan[task].arguments
    assert "--quick" in plan["ledger_build"].arguments
    assert "--quick" in plan["analyse"].arguments
    assert "--quick" not in plan["ingest"].arguments


def test_the_feature_selector_reaches_dbt_unexpanded_by_the_shell() -> None:
    plan = {step.task_id: step for step in pipeline.steps("public", quick=False, synthetic_dir="s")}
    assert pipeline.command(plan["features"], "public").endswith("--select 'features_*'")
    assert pipeline.command(plan["dbt_run"], "public").endswith("--exclude 'features_*'")


def test_an_unknown_zone_is_refused() -> None:
    with pytest.raises(ValueError, match="zone"):
        pipeline.steps("staging", quick=False, synthetic_dir="s")


@pytest.mark.parametrize(
    "step",
    [s for s in pipeline.steps("public", quick=True, synthetic_dir="s") if s.arguments[0] != "dbt"],
    ids=lambda step: step.task_id,
)
def test_every_task_is_a_strata_command_that_parses(step: Any) -> None:
    result = CliRunner().invoke(cli.app, [*step.arguments, "--help"])
    assert result.exit_code == 0, result.output


class _Operator:
    def __init__(self, *, task_id: str, bash_command: str, doc_md: str | None = None) -> None:
        self.task_id = task_id
        self.bash_command = bash_command
        self.doc_md = doc_md
        self.downstream: set[str] = set()

    def __rshift__(self, other: _Operator) -> _Operator:
        self.downstream.add(other.task_id)
        return other


class _Dag:
    built: ClassVar[list[_Dag]] = []

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        _Dag.built.append(self)

    def __enter__(self) -> _Dag:
        return self

    def __exit__(self, *exc: object) -> None:
        return None


def _stand_in_for_airflow(monkeypatch: pytest.MonkeyPatch) -> None:
    names = [
        "airflow",
        "airflow.sdk",
        "airflow.providers",
        "airflow.providers.standard",
        "airflow.providers.standard.operators",
        "airflow.providers.standard.operators.bash",
    ]
    modules = {name: types.ModuleType(name) for name in names}
    modules["airflow.sdk"].DAG = _Dag  # type: ignore[attr-defined]
    modules["airflow.providers.standard.operators.bash"].BashOperator = _Operator  # type: ignore[attr-defined]
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)


# Ids without the word "private": that keyword marks the tests that need the real extract.
@pytest.mark.parametrize("zone", ["public", "private"], ids=["synthetic-zone", "real-data-zone"])
def test_the_dag_file_wires_the_plan(monkeypatch: pytest.MonkeyPatch, zone: str) -> None:
    _stand_in_for_airflow(monkeypatch)
    monkeypatch.setenv("STRATA_DAG_ZONE", zone)
    monkeypatch.setenv("STRATA_DAG_EFFORT", "quick")
    monkeypatch.setenv("STRATA_BIN", "/opt/strata/bin/strata")
    module = _load("strata_daily")
    dag = _Dag.built[-1]
    assert dag.kwargs["dag_id"] == "strata_daily"
    assert dag.kwargs["max_active_tasks"] == 1
    assert dag.kwargs["catchup"] is False
    plan = pipeline.steps(zone, quick=True, synthetic_dir=module.SYNTH_DIR)
    tasks: dict[str, _Operator] = module.tasks
    assert list(tasks) == _ids(plan)
    for step in plan:
        assert tasks[step.task_id].bash_command == pipeline.command(
            step, zone, "/opt/strata/bin/strata"
        )
        for upstream in step.upstream:
            assert step.task_id in tasks[upstream].downstream
    flagged = all("--private" in task.bash_command for task in tasks.values())
    assert flagged is (zone == "private")
