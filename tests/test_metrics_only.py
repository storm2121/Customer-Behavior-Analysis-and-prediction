"""Model-quality claims and their ``metrics_only`` publication: the allowlist, the build, the
renderer, the check and the guard."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from strata.config import Settings
from strata.ledger.allowlist import (
    METRICS_ONLY_IDS,
    METRICS_ONLY_KEYS,
    ledger_violations,
    public_violations,
)
from strata.ledger.build import PUBLIC_LEDGER, _publish, metrics_view
from strata.ledger.model import ClaimRecord, Ledger, Unit
from strata.ledger.registry import ClaimContext, ClaimError
from strata.ledger.render import Placeholders
from strata.privacy_guard import Scanner, disk_files, scan


def _full(record: ClaimRecord) -> ClaimRecord:
    """The record as a full-effort private build holds it (the synthetic fixture is quick)."""
    meta = {key: text for key, text in record.meta.items() if key != "effort"}
    return record.model_copy(
        update={"meta": {**meta, "model_effort": "full"}, "source_kind": "real"}
    )


def _published(synthetic_ledger: Ledger, claim_id: str = "silence_model_quality") -> dict[str, Any]:
    return metrics_view(_full(synthetic_ledger.get(claim_id))).model_dump(mode="json")


def test_two_model_quality_claims_are_allow_listed_for_their_metrics_only() -> None:
    assert {"silence_model_quality", "segment_stability"} == METRICS_ONLY_IDS
    assert METRICS_ONLY_KEYS["silence_model_quality"] == {
        "brier",
        "calibration_error",
        "lift_at_10",
    }
    assert METRICS_ONLY_KEYS["segment_stability"] == {"mean_ari"}


def test_the_claims_read_the_models_trained_on_this_warehouse(synthetic_ledger: Ledger) -> None:
    quality = synthetic_ledger.get("silence_model_quality")
    assert set(quality.values) == {"brier", "calibration_error", "lift_at_10"}
    assert set(quality.ci) == set(quality.values)
    assert quality.units["lift_at_10"] is Unit.RATIO
    assert quality.meta["model"] == "lightgbm_platt"
    assert quality.meta["model_effort"] == "quick"
    assert quality.n > 0
    assert set(quality.value_n.values()) == {quality.n}
    stability = synthetic_ledger.get("segment_stability")
    assert set(stability.values) == {"mean_ari"}
    low, high = stability.ci["mean_ari"]
    assert -1 <= low <= high <= 1
    assert stability.interval_label == "95% range of refits"


def test_the_metrics_view_keeps_the_metrics_and_drops_everything_else(
    synthetic_ledger: Ledger,
) -> None:
    record = _full(synthetic_ledger.get("silence_model_quality"))
    noisy = record.model_copy(
        update={
            "values": {**record.values, "shap_recency_days": 0.3},
            "units": {**record.units, "shap_recency_days": Unit.NUMBER},
            "meta": {**record.meta, "rows_top_clients": "a:b"},
        }
    )
    view = metrics_view(noisy)
    assert set(view.values) == METRICS_ONLY_KEYS["silence_model_quality"]
    assert set(view.units) == set(view.values)
    assert "rows_top_clients" not in view.meta
    assert view.robustness is None
    assert view.public
    assert public_violations([view.model_dump(mode="json")]) == []


def test_quick_models_and_small_samples_are_not_published(synthetic_ledger: Ledger) -> None:
    with pytest.raises(ClaimError, match="quick"):
        metrics_view(synthetic_ledger.get("silence_model_quality"))
    small = _full(synthetic_ledger.get("silence_model_quality")).model_copy(update={"n": 10})
    with pytest.raises(ClaimError, match="floor"):
        metrics_view(small)
    with pytest.raises(ClaimError, match="not allow-listed"):
        metrics_view(_full(synthetic_ledger.get("channel_mix")))


def test_the_rules_refuse_anything_beyond_the_metrics(synthetic_ledger: Ledger) -> None:
    view = _published(synthetic_ledger)
    assert public_violations([view]) == []
    importance = {**view, "values": {**view["values"], "shap_recency_days": 0.3}}
    assert any("beyond its model-quality metrics" in p for p in public_violations([importance]))
    robust = {**view, "robustness": {"verdict": "holds"}}
    assert any("robustness" in p for p in public_violations([robust]))
    extra = {**view, "meta": {**view["meta"], "top_client": "anyone"}}
    assert any("beyond how its metrics" in p for p in public_violations([extra]))
    quick = {**view, "meta": {**view["meta"], "model_effort": "quick"}}
    assert any("full effort" in p for p in public_violations([quick]))
    small = {**view, "n": 20}
    assert any("fewer than" in p for p in public_violations([small]))
    empty = {**view, "values": {}, "units": {}, "ci": {}, "value_n": {}}
    assert any("no metric" in p for p in public_violations([empty]))
    assert ledger_violations({"claims": [importance], "verdicts": []})


def test_the_guard_refuses_a_metrics_claim_carrying_an_importance(
    synthetic_ledger: Ledger, git_repo: Path
) -> None:
    view = _published(synthetic_ledger)
    view["values"]["shap_recency_days"] = 0.3
    path = git_repo / "docs" / "ledger" / "claims.public.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"claims": [view], "verdicts": []}), encoding="utf-8")
    findings = scan(disk_files([path], git_repo), Scanner.from_environment(environ={}))
    assert any(finding.check == "ledger-allowlist" for finding in findings)


def test_placeholders_render_the_metrics_and_refuse_the_rest(synthetic_ledger: Ledger) -> None:
    view = metrics_view(_full(synthetic_ledger.get("silence_model_quality")))
    public = Ledger(
        source_kind="real",
        data_fingerprint=synthetic_ledger.data_fingerprint,
        computed_at=synthetic_ledger.computed_at,
        claims=[view],
    )
    places = Placeholders(public)
    assert places.value("silence_model_quality", "lift_at_10").endswith("\N{MULTIPLICATION SIGN}")
    assert " to " in places.interval("silence_model_quality", "brier")
    assert places.raw("silence_model_quality", "calibration_error") >= 0
    with pytest.raises(KeyError, match="model-quality metrics only"):
        places.value("silence_model_quality", "auc")
    with pytest.raises(KeyError, match="model-quality metrics only"):
        places.verdict("silence_model_quality")
    with pytest.raises(KeyError, match="model-quality metrics only"):
        places.mechanism("silence_model_quality")
    with pytest.raises(KeyError, match="no tables"):
        places.rows("silence_model_quality", "scores")


def test_the_public_ledger_carries_the_metrics_view(
    synthetic_ledger: Ledger, tmp_path: Path
) -> None:
    settings = Settings(_env_file=None, repo_root=tmp_path / "repo")  # type: ignore[call-arg]
    full = synthetic_ledger.get("channel_mix").model_copy(update={"source_kind": "real"})
    noisy = _full(synthetic_ledger.get("segment_stability"))
    noisy = noisy.model_copy(update={"meta": {**noisy.meta, "assignment_file": "x"}})
    private = Ledger(
        source_kind="real",
        data_fingerprint=synthetic_ledger.data_fingerprint,
        computed_at=synthetic_ledger.computed_at,
        claims=[full, noisy],
    )
    path = _publish(settings, private, skipped={})
    assert path == settings.repo_root / PUBLIC_LEDGER
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert [record["id"] for record in payload["claims"]] == ["channel_mix", "segment_stability"]
    stability = payload["claims"][1]
    assert set(stability["values"]) == {"mean_ari"}
    assert set(stability["meta"]) == {"model_effort"}
    assert ledger_violations(payload) == []
    with pytest.raises(ClaimError, match="could not be computed"):
        _publish(settings, private, skipped={"silence_model_quality": "not trained"})


def test_a_model_trained_on_other_data_is_a_skipped_claim(tmp_path: Path) -> None:
    models = tmp_path / "models"
    models.mkdir()
    entry = {
        "name": "silence",
        "data_fingerprint": "a" * 64,
        "mapping_sha256": None,
        "results_file": "silence/results.json",
        "effort": "full",
    }
    (models / "registry.json").write_text(json.dumps({"models": [entry]}), encoding="utf-8")
    context = ClaimContext(None, models_dir=models, fingerprint="b" * 64)  # type: ignore[arg-type]
    with pytest.raises(ClaimError, match="other data"):
        context.model_results("silence")
    with pytest.raises(ClaimError, match="not trained"):
        context.model_results("segments")
    same = ClaimContext(None, models_dir=models, fingerprint="a" * 64)  # type: ignore[arg-type]
    with pytest.raises(ClaimError, match="no results file"):
        same.model_results("silence")
    empty = ClaimContext(None, models_dir=tmp_path / "none")  # type: ignore[arg-type]
    with pytest.raises(ClaimError, match="no trained models"):
        empty.model_results("silence")
    with pytest.raises(ClaimError, match="no model folder"):
        ClaimContext(None).model_results("silence")  # type: ignore[arg-type]
