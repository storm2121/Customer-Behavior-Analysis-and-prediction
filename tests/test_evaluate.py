"""The evaluation gate: models trained on the current data and better than chance."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import pytest
from typer.testing import CliRunner

from strata import cli
from strata.ingest.manifest import Manifest
from strata.models import evaluate, registry

FINGERPRINT = "f" * 64
MAPPING = "m" * 64


def _metric(low: float) -> dict[str, Any]:
    return {"value": low + 0.05, "ci": [low, low + 0.1]}


HEADLINES: dict[str, dict[str, Any]] = {
    "silence": {"metrics": {"lightgbm_platt": {"auc": _metric(0.8)}}},
    "survival": {"concordance": 0.7},
    "adoption": {
        "targets": {
            "target_has_online": {"metrics": {"lightgbm": {"auc": _metric(0.7)}}},
            "target_active_online": {"metrics": {"lightgbm": {"auc": _metric(0.6)}}},
        }
    },
    "segments": {"stability": {"mean_ari": 0.8, "ci_low": 0.7, "ci_high": 0.9}},
}


def _registry(**overrides: dict[str, Any]) -> registry.Registry:
    entries = [
        registry.entry(
            name=name,
            title=name,
            manifest_fingerprint=FINGERPRINT,
            mapping_sha256=MAPPING,
            source_kind="synthetic",
            effort="quick",
            params={},
            headline=overrides.get(name, headline),
            run_id=None,
            results_file=f"{name}/results.json",
        )
        for name, headline in HEADLINES.items()
    ]
    return registry.Registry(zone="public", source_kind="synthetic", models=entries)


def _manifest(fingerprint: str = FINGERPRINT, mapping: str = MAPPING) -> Manifest:
    return cast("Manifest", SimpleNamespace(fingerprint=fingerprint, mapping_sha256=mapping))


def _failed(checks: list[evaluate.Check]) -> list[tuple[str, str]]:
    return [(check.model, check.check) for check in checks if not check.passed]


def test_current_models_better_than_chance_pass() -> None:
    checks = evaluate.evaluate(_registry(), _manifest())
    assert _failed(checks) == []
    assert {check.model for check in checks} == set(HEADLINES)


def test_models_trained_on_other_data_or_another_mapping_fail() -> None:
    assert ("silence", "trained on the data in the warehouse") in _failed(
        evaluate.evaluate(_registry(), _manifest(fingerprint="0" * 64))
    )
    assert len(_failed(evaluate.evaluate(_registry(), _manifest(mapping="0" * 64)))) == len(
        HEADLINES
    )


@pytest.mark.parametrize(
    ("name", "headline"),
    [
        ("silence", {"metrics": {"lightgbm_platt": {"auc": _metric(0.45)}}}),
        ("silence", {"metrics": {}}),
        ("survival", {"concordance": 0.5}),
        ("survival", {"concordance": float("nan")}),
        (
            "adoption",
            {"targets": {"target_has_online": {"metrics": {"lightgbm": {"auc": _metric(0.4)}}}}},
        ),
        ("adoption", {"targets": {}}),
        ("segments", {"stability": {"mean_ari": -0.1}}),
    ],
)
def test_a_model_below_its_floor_fails(name: str, headline: dict[str, Any]) -> None:
    failed = _failed(evaluate.evaluate(_registry(**{name: headline}), _manifest()))
    assert [model for model, _ in failed] == [name]


def test_a_missing_model_fails() -> None:
    models = _registry()
    partial = registry.Registry(
        zone="public",
        source_kind="synthetic",
        models=[m for m in models.models if m.name != "survival"],
    )
    assert _failed(evaluate.evaluate(partial, _manifest())) == [("survival", "in the registry")]


def test_checks_name_the_model_and_the_check_never_a_value() -> None:
    labels = [check.check for check in evaluate.evaluate(_registry(), _manifest())]
    assert not any(character.isdigit() for label in labels for character in label)


def test_the_command_exits_non_zero_when_a_check_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    failing = [evaluate.Check("silence", "served score ranks better than chance", passed=False)]
    monkeypatch.setattr(evaluate, "evaluate_zone", lambda settings, zone: failing)
    result = CliRunner().invoke(cli.app, ["evaluate"])
    assert result.exit_code == 1
    assert "FAIL" in result.output


def test_the_command_passes_when_every_check_passes(monkeypatch: pytest.MonkeyPatch) -> None:
    passing = [evaluate.Check("silence", "served score ranks better than chance", passed=True)]
    monkeypatch.setattr(evaluate, "evaluate_zone", lambda settings, zone: passing)
    result = CliRunner().invoke(cli.app, ["evaluate"])
    assert result.exit_code == 0
    assert "pass" in result.output
