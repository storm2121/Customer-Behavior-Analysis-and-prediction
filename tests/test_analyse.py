"""``strata analyse``: modules recompute their claims, draw figures and run the simulator."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import numpy as np
import pytest
from typer.testing import CliRunner

from strata.analysis import figures, markov
from strata.analysis.run import MODULES, SIMULATOR_FILE, run_module
from strata.cli import app
from strata.config import Settings, Zone, is_within
from strata.ledger.build import load_ledger
from strata.ledger.registry import registered

if TYPE_CHECKING:
    from pathlib import Path

    from strata.ledger.model import Ledger


def test_every_module_claim_is_registered_once() -> None:
    claimed = [claim_id for module in MODULES.values() for claim_id in module.claims]
    assert len(claimed) == len(set(claimed))
    assert set(claimed) <= set(registered())


def test_a_module_recomputes_its_claims_and_draws_its_figures(built_warehouse: Settings) -> None:
    result = run_module(built_warehouse, Zone.PUBLIC, "concentration", quick=True)
    workspace = built_warehouse.workspace(Zone.PUBLIC)
    assert result.claims == ["spend_concentration", "rfm_by_channel"]
    assert result.skipped == {}
    assert [path.name for path in result.outputs] == ["concentration_lorenz.html"]
    assert all(is_within(path, workspace.figures_dir) for path in result.outputs)
    assert (workspace.figures_dir / "plotly.min.js").is_file()
    ledger = load_ledger(workspace.ledger_dir / "claims.synthetic.json")
    assert {"spend_concentration", "rfm_by_channel"} <= set(ledger.ids)
    assert ledger.get("rfm_by_channel").meta["effort"] == "quick"


def test_the_markov_module_saves_what_the_simulator_needs(built_warehouse: Settings) -> None:
    result = run_module(built_warehouse, Zone.PUBLIC, "markov", quick=True)
    assert result.policy is not None
    path = built_warehouse.workspace(Zone.PUBLIC).root / SIMULATOR_FILE
    assert path in result.outputs
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["states"] == list(markov.STATES)
    assert np.asarray(payload["transition_counts"]).shape == (4, 4)
    assert sum(payload["start_counts"]) == pytest.approx(
        sum(payload["sample_policy"]["start_counts"])
    )
    bands = payload["sample_policy"]["bands"]
    for quantity in markov.QUANTITIES:
        for low, middle, high in zip(
            bands[quantity]["p05"], bands[quantity]["p50"], bands[quantity]["p95"], strict=True
        ):
            assert low <= middle <= high
    assert len(bands["digital_share"]["p50"]) == 13


def test_unknown_modules_are_refused(built_warehouse: Settings) -> None:
    with pytest.raises(ValueError, match="unknown module"):
        run_module(built_warehouse, Zone.PUBLIC, "astrology")


def _cli(settings: Settings, monkeypatch: pytest.MonkeyPatch, *args: str) -> tuple[int, str]:
    monkeypatch.setenv("STRATA_DB", str(settings.db))
    monkeypatch.setenv("STRATA_WORK_DIR", str(settings.work_dir))
    result = CliRunner().invoke(app, list(args))
    return result.exit_code, result.output


def test_cli_analyse_prints_the_sample_policy_bands(
    built_warehouse: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    code, output = _cli(built_warehouse, monkeypatch, "analyse", "markov", "--quick")
    assert code == 0, output
    assert "what if 30% of branch-only clients under 45 move to mixed use" in output
    assert "analyse (public, quick)" in output
    assert "channel_markov" in output


def test_cli_analyse_runs_a_module_with_figures(
    built_warehouse: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    code, output = _cli(built_warehouse, monkeypatch, "analyse", "benford", "--quick")
    assert code == 0, output
    assert "benford_by_category" in output
    figure = built_warehouse.workspace(Zone.PUBLIC).figures_dir / "benford_first_digits.html"
    assert figure.is_file()


def test_cli_analyse_refuses_unknown_modules(
    built_warehouse: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    code, output = _cli(built_warehouse, monkeypatch, "analyse", "astrology")
    assert code == 2
    assert "choose all or one of" in output


def test_the_seasonality_module_draws_one_figure_per_channel(built_warehouse: Settings) -> None:
    result = run_module(built_warehouse, Zone.PUBLIC, "seasonality", quick=True)
    names = sorted(path.name for path in result.outputs)
    assert names == [
        "seasonality_stl_branch.html",
        "seasonality_stl_card.html",
        "seasonality_stl_online.html",
    ]


def test_figures_draw_from_ledger_records(synthetic_ledger: Ledger, tmp_path: Path) -> None:
    record = synthetic_ledger.get("calendar_effects")
    forest = figures.calendar_forest(record)
    assert len(forest.data[0].x) == len([key for key in record.values if key in record.ci])
    path = figures.write(forest, tmp_path / "figures", "calendar_effects")
    assert path.read_text(encoding="utf-8").startswith("<!doctype html>")
    assert (tmp_path / "figures" / "plotly.min.js").is_file()


def test_the_simulation_fan_shows_policy_and_baseline() -> None:
    samples = markov.posterior_samples(np.full((4, 4), 10.0), n_samples=50, seed=0)
    rates = {"none": 0.0, "branch_only": 2.0, "digital_only": 0.0, "mixed": 1.0}
    start = np.array([10.0, 30.0, 20.0, 40.0])
    moved = start + np.array([0.0, -9.0, 0.0, 9.0])
    fig = figures.simulation_fan(
        markov.simulate_paths(samples, moved, rates),
        markov.simulate_paths(samples, start, rates),
        "a test policy",
    )
    names = {trace.name for trace in fig.data if trace.name}
    assert names == {"baseline", "policy"}
    assert "What if a test policy?" in fig.layout.title.text
