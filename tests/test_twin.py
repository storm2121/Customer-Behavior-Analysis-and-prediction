"""The synthetic twin: its refusals first, then the private copula, its sampling and its metrics.

Every frame here is generated in the test; the twin of the real clients is built only by
`strata --private synth-twin` on the machine that holds the extract.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import duckdb
import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from strata import cli
from strata.config import PrivacyBoundaryError, Settings, Zone
from strata.synthetic import twin

NAMES = [column.name for column in twin.COLUMNS]


def _settings(tmp_path: Path) -> Settings:
    repo = tmp_path / "repo"
    repo.mkdir(exist_ok=True)
    return Settings(  # type: ignore[call-arg]
        _env_file=None,
        repo_root=repo,
        db=repo / ".strata" / "strata.duckdb",
        work_dir=repo / ".strata",
        data_dir=tmp_path / "extract",
        private_dir=tmp_path / "private",
    )


def _clients(n: int = 3000, seed: int = 0) -> pd.DataFrame:
    """Made-up clients with the twin's columns and some structure between them."""
    rng = np.random.default_rng(seed)
    age = rng.integers(18, 90, n).astype(float)
    age[rng.random(n) < 0.05] = np.nan
    young = np.nan_to_num(age, nan=50) < 45
    activity = rng.gamma(2.0, 20.0, n)
    n_card = rng.poisson(activity * np.where(young, 1.0, 0.3))
    n_online = rng.poisson(activity * np.where(young, 0.6, 0.1))
    n_branch = rng.poisson(activity * np.where(young, 0.2, 0.8))
    n_events = n_card + n_online + n_branch
    digital = (n_card + n_online) > 0
    branch = n_branch > 0
    profile = np.select(
        [digital & branch, digital, branch], ["mixed", "digital_only", "branch_only"], "none"
    )
    return pd.DataFrame(
        {
            "age": pd.array(age, dtype="Float64"),
            "tenure_years": rng.integers(0, 40, n),
            "gender": rng.choice(["gender_1", "gender_2"], n),
            "marital_status": rng.choice(["marital_1", "marital_2", "marital_3", "unexpected"], n),
            "channel_profile": profile,
            "active_months": np.minimum(12, rng.poisson(6, n)),
            "n_events": n_events,
            "n_card": n_card,
            "n_online": n_online,
            "n_branch": n_branch,
            "card_spend": n_card * rng.lognormal(6, 1, n),
            "branch_credit_amount": n_branch * rng.lognormal(8, 1, n),
            "online_amount": n_online * rng.lognormal(5, 1, n),
            "branch_share": np.where(n_events > 0, n_branch / np.maximum(n_events, 1), np.nan),
        }
    )


# ---------------------------------------------------------------------------------- refusals


def test_a_destination_inside_the_repository_is_refused_before_anything_is_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("CI", raising=False)
    settings = _settings(tmp_path)
    for destination in (
        settings.repo_root / "twin",
        settings.repo_root / "docs" / ".." / "twin",
        settings.repo_root,
    ):
        for zone in (Zone.PRIVATE, Zone.PUBLIC):
            with pytest.raises(PrivacyBoundaryError, match="inside the repository"):
                twin.run(settings, zone, destination, twin.TwinConfig())
    assert not (settings.repo_root / "twin").exists()


def test_the_command_refuses_a_destination_inside_the_repository(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("CI", raising=False)
    monkeypatch.setenv("STRATA_DATA_DIR", str(tmp_path / "extract"))
    monkeypatch.setenv("STRATA_PRIVATE_DIR", str(tmp_path / "private"))
    inside = Settings().repo_root / ".strata" / "twin-must-not-exist"
    result = CliRunner().invoke(cli.app, ["--private", "synth-twin", "--out", str(inside)])
    assert result.exit_code != 0
    # The command raises; `strata`'s entry point turns the error into a one-line report.
    assert isinstance(result.exception, PrivacyBoundaryError)
    assert "inside the repository" in str(result.exception)
    assert not inside.exists()


def test_the_twin_is_refused_in_ci(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CI", "true")
    settings = _settings(tmp_path)
    with pytest.raises(PrivacyBoundaryError, match="never in CI"):
        twin.run(settings, Zone.PRIVATE, tmp_path / "private" / "twin", twin.TwinConfig())


def test_a_destination_outside_the_private_folder_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("CI", raising=False)
    with pytest.raises(PrivacyBoundaryError, match="STRATA_PRIVATE_DIR"):
        twin.run(_settings(tmp_path), Zone.PRIVATE, tmp_path / "elsewhere", twin.TwinConfig())


def test_the_public_zone_has_no_twin(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CI", raising=False)
    with pytest.raises(twin.TwinError, match="--private"):
        twin.run(_settings(tmp_path), Zone.PUBLIC, tmp_path / "private" / "twin", twin.TwinConfig())


@pytest.mark.parametrize(
    "arguments",
    [{"epsilon": 0.0}, {"epsilon": 101.0}, {"marginal_share": 1.0}, {"holdout": 0.0}],
)
def test_a_meaningless_configuration_is_refused(arguments: dict[str, float]) -> None:
    with pytest.raises(twin.TwinError):
        twin.TwinConfig(**arguments)  # type: ignore[arg-type]


def test_too_few_clients_are_refused() -> None:
    with pytest.raises(twin.TwinError, match="at least"):
        twin.build(_clients(n=100), twin.TwinConfig())


# ---------------------------------------------------------------------------------- pieces


def test_kendall_tau_a_counts_ties_as_neither_concordant_nor_discordant() -> None:
    x = np.array([1.0, 2.0, 2.0, 3.0, 4.0])
    y = np.array([1.0, 3.0, 2.0, 2.0, 5.0])
    concordant = discordant = 0
    for i in range(len(x)):
        for j in range(i + 1, len(x)):
            sign = np.sign(x[i] - x[j]) * np.sign(y[i] - y[j])
            concordant += int(sign > 0)
            discordant += int(sign < 0)
    expected = (concordant - discordant) / (len(x) * (len(x) - 1) / 2)
    assert twin.kendall_tau_a(x, y) == pytest.approx(expected)
    assert twin.kendall_tau_a(np.ones(5), y) == 0.0
    assert twin.tau_sensitivity(1001) == pytest.approx(4 / 1000)


def test_the_nearest_correlation_matrix_is_positive_with_a_unit_diagonal() -> None:
    indefinite = np.array([[1.0, 0.9, -0.9], [0.9, 1.0, 0.9], [-0.9, 0.9, 1.0]])
    fixed = twin.nearest_correlation(indefinite)
    assert np.allclose(np.diag(fixed), 1.0)
    assert np.linalg.eigvalsh(fixed).min() > 0
    np.linalg.cholesky(fixed)


def test_bins_come_from_public_bounds_and_catch_everything_else() -> None:
    age = next(column for column in twin.COLUMNS if column.name == "age")
    index = twin.bin_index(age, pd.Series([18, 17, 150, None], dtype="Float64"))
    assert index.tolist() == [0, 0, age.n_bins() - 2, age.n_bins() - 1]
    gender = next(column for column in twin.COLUMNS if column.name == "gender")
    codes = twin.bin_index(gender, pd.Series(["gender_2", "unknown", None], dtype="string"))
    assert codes.tolist() == [1, len(gender.domain), len(gender.domain) + 1]


# ---------------------------------------------------------------------------------- the twin


@pytest.fixture(scope="module")
def built() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    frame = _clients()
    result, summary = twin.build(frame, twin.TwinConfig(epsilon=1.0, seed=3))
    return frame, result, summary


def test_the_twin_has_the_columns_and_only_values_its_bins_allow(
    built: tuple[pd.DataFrame, pd.DataFrame, dict[str, object]],
) -> None:
    frame, result, summary = built
    assert list(result.columns) == NAMES
    assert len(result) == summary["n_twin"] == 1000
    assert summary["n_train"] == round(len(frame) * 0.8)
    assert result["age"].dropna().between(18, 101).all()
    assert result["branch_share"].dropna().between(0, 1).all()
    assert set(result["gender"].dropna()) <= {"gender_1", "gender_2", twin.OTHER}
    assert set(result["channel_profile"].dropna()) <= {
        "none",
        "branch_only",
        "digital_only",
        "mixed",
        twin.OTHER,
    }
    assert (result["n_events"].dropna() % 1 == 0).all()


def test_the_twin_keeps_the_dependence_between_columns_when_the_budget_allows() -> None:
    # With a few thousand clients and fourteen columns, a budget of one drowns the ninety-one
    # pairwise correlations in noise; a large budget shows the copula itself keeps them.
    frame = _clients(seed=7)
    result, _ = twin.build(frame, twin.TwinConfig(epsilon=50.0, seed=2))
    real = frame["n_events"].corr(frame["n_card"], method="spearman")
    copy = result["n_events"].astype(float).corr(result["n_card"].astype(float), method="spearman")
    assert real > 0.5
    assert copy > 0.5 * real


def test_the_metrics_are_complete_and_in_range(
    built: tuple[pd.DataFrame, pd.DataFrame, dict[str, object]],
) -> None:
    _, _, summary = built
    fidelity = summary["fidelity"]
    privacy = summary["privacy"]
    assert isinstance(fidelity, dict)
    assert isinstance(privacy, dict)
    assert set(fidelity["marginal_distance"]) == set(NAMES)
    assert all(0 <= value <= 1 for value in fidelity["marginal_distance"].values())
    assert 0 <= fidelity["distinguishability_auc"] <= 1
    assert privacy["dcr_ratio"] > 0
    for key in ("membership_inference_auc", "share_twin_closer_than_holdout_p05"):
        assert 0 <= privacy[key] <= 1
    assert 0 <= privacy["membership_inference_advantage"] <= 1
    assert summary["epsilon_marginals"] + summary["epsilon_correlations"] == pytest.approx(1.0)


def test_a_large_budget_reproduces_the_marginals_closely() -> None:
    frame = _clients(seed=5)
    _, summary = twin.build(frame, twin.TwinConfig(epsilon=100.0, seed=1))
    distances = summary["fidelity"]["marginal_distance"]  # type: ignore[index]
    assert distances["gender"] < 0.05
    assert distances["age"] < 0.05


def test_the_same_seed_gives_the_same_twin() -> None:
    frame = _clients(n=600, seed=9)
    first, _ = twin.build(frame, twin.TwinConfig(seed=4))
    second, _ = twin.build(frame, twin.TwinConfig(seed=4))
    pd.testing.assert_frame_equal(first, second)


def test_the_command_writes_the_twin_and_its_metrics_under_the_private_folder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("CI", raising=False)
    settings = _settings(tmp_path)
    workspace = settings.workspace(Zone.PRIVATE)
    workspace.warehouse.parent.mkdir(parents=True)
    con = duckdb.connect(str(workspace.warehouse))
    clients = _clients(n=800, seed=2)
    clients.insert(0, "client_id", [f"c{i:05d}" for i in range(len(clients))])
    con.execute("create schema marts")
    con.register("frame", clients)
    con.execute("create table marts.client_360 as select * from frame")
    con.close()
    manifest = SimpleNamespace(fingerprint="f" * 64, mapping_sha256=None)
    monkeypatch.setattr(twin, "load_manifest", lambda _workspace: manifest)
    written = twin.run(settings, Zone.PRIVATE, tmp_path / "private" / "twin", twin.TwinConfig())
    assert [path.name for path in written] == [twin.TWIN_FILE, twin.METRICS_FILE]
    assert all(path.is_relative_to(tmp_path / "private") for path in written)
    rows = pd.read_csv(written[0])
    assert "client_id" not in rows.columns
    assert list(rows.columns) == NAMES
    summary = json.loads(written[1].read_text(encoding="utf-8"))
    assert summary["data_fingerprint"] == "f" * 64
    assert {"fidelity", "privacy", "epsilon"} <= set(summary)
    assert not (settings.repo_root / twin.TWIN_FILE).exists()
