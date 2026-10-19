"""``strata train``: fit the risk models on the zone's warehouse and register them.

Results, scores and the zone's registry go to ``<workspace>/models`` (under ``STRATA_PRIVATE_DIR``
on a private run), every run is logged to the zone's MLflow store, and model cards are rendered
next to the registry. ``--promote`` (public zone only) then publishes the registry and the cards
into the repository: ``models/registry.json`` and ``docs/model_cards/``.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from strata.config import Settings, Workspace, Zone
from strata.ingest.load import load_manifest
from strata.ingest.manifest import Manifest
from strata.models import adoption, cards, churn, plots, registry, segments, survival
from strata.warehouse import connect, read_relation

TITLES = {
    "silence": "Silence risk in the next three months",
    "survival": "Time between active days and time to silence",
    "adoption": "Online banking adoption (aggregate sample)",
    "segments": "Behavioural segments of active clients",
}
MODELS = tuple(TITLES)


def _json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _top(mapping: dict[str, float], count: int) -> dict[str, float]:
    return dict(list(mapping.items())[:count])


@dataclass(frozen=True)
class Context:
    settings: Settings
    zone: Zone
    workspace: Workspace
    manifest: Manifest
    quick: bool
    seed: int

    @property
    def folder(self) -> Path:
        return self.workspace.root / "models"

    def read(self, name: str) -> pd.DataFrame:
        con = connect(self.workspace.warehouse, read_only=True)
        try:
            return read_relation(con, name)
        finally:
            con.close()


@dataclass(frozen=True)
class Trained:
    params: dict[str, Any]
    headline: dict[str, Any]
    results: dict[str, Any]
    tables: dict[str, pd.DataFrame]


def _silence(context: Context) -> Trained:
    effort = churn.QUICK if context.quick else churn.FULL
    result = churn.run(context.read("marts.features_silence"), effort=effort, seed=context.seed)
    primary = result.primary.as_dict()
    odds = sorted(result.odds_ratios.items(), key=lambda item: -abs(np.log(item[1]["odds_ratio"])))
    headline = {
        **{k: primary[k] for k in ("train_cutoff", "test_cutoff", "n_train", "n_test")},
        "silence_rate_train": primary["silence_rate_train"],
        "silence_rate_test": primary["silence_rate_test"],
        "metrics": primary["metrics"],
        "auc_gain_over_glm": result.auc_gain_over_glm.as_dict(),
        "backtests": [
            {
                "train_cutoff": b.train_cutoff,
                "test_cutoff": b.test_cutoff,
                "auc": b.metrics[churn.SERVED]["auc"].as_dict(),
                "lift_at_10": b.metrics[churn.SERVED]["lift_at_10"].as_dict(),
                "shap_top": _top(b.shap_global, 3),
            }
            for b in result.backtests
        ],
        "shap_top": _top(result.shap_global, 8),
        "odds_ratios_top": dict(odds[:8]),
    }
    results = {
        "params": result.params,
        "backtests": [b.as_dict() for b in result.backtests],
        "odds_ratios": result.odds_ratios,
        "shap_global": result.shap_global,
    }
    return Trained(
        params={**result.params, "trials": effort.trials, "resamples": effort.resamples},
        headline=headline,
        results=results,
        tables={"scored": result.scored},
    )


def _survival(context: Context) -> Trained:
    resamples = 50 if context.quick else 200
    result = survival.run(
        context.read("intermediate.int_events"),
        context.read("marts.client_360"),
        n_resamples=resamples,
        seed=context.seed,
    )
    headline = {
        "gap_by_profile": result["gap_by_profile"],
        "logrank_p": result["gap_logrank_profile"]["p_value"],
        "hazard_ratios": result["cox"]["hazard_ratios"],
        "ph_test_p_values": result["cox"]["ph_test_p_values"],
        "concordance": result["cox"]["concordance"],
    }
    return Trained({"resamples": resamples}, headline, result, {})


def _adoption(context: Context) -> Trained:
    resamples = 100 if context.quick else 1000
    result = adoption.run(
        context.read("marts.aggregate_clients"), n_resamples=resamples, seed=context.seed
    )
    headline = {
        "targets": {
            target: {
                "n_clients": item["n_clients"],
                "rate": item["rate"],
                "metrics": item["metrics"],
                "importance_top": _top(item["importance"], 6),
            }
            for target, item in result.items()
        }
    }
    return Trained({"resamples": resamples, "folds": 5}, headline, result, {})


def _segments(context: Context) -> Trained:
    effort = segments.QUICK if context.quick else segments.FULL
    result = segments.run(context.read("marts.features_segments"), effort=effort, seed=context.seed)
    assignments = result.pop("assignments")
    headline = {
        "k": result["k"],
        "stability": result["stability"],
        "segments": [
            {key: s[key] for key in ("segment", "name", "n_clients", "share")}
            for s in result["segments"]
        ],
        "bic": result["bic"],
        "silhouette": result["silhouette"],
    }
    params = {"k_max": effort.k_max, "resamples": effort.resamples, "umap": effort.umap}
    return Trained(params, headline, result, {"assignments": assignments})


TRAINERS: dict[str, Callable[[Context], Trained]] = {
    "silence": _silence,
    "survival": _survival,
    "adoption": _adoption,
    "segments": _segments,
}


def _write_table(frame: pd.DataFrame, path: Path) -> None:
    records = json.loads(frame.to_json(orient="records", date_format="iso"))
    path.write_text(json.dumps(records, indent=1) + "\n", encoding="utf-8", newline="\n")


def train(
    settings: Settings, zone: Zone, names: list[str], *, quick: bool = False
) -> registry.Registry:
    """Train ``names`` (every model by default), log them and update the zone's registry."""
    workspace = settings.workspace(zone)
    if workspace.is_private:
        settings.assert_private_output(workspace.root / "models")
        settings.assert_private_output(workspace.root / "mlruns")
    manifest = load_manifest(workspace)
    context = Context(settings, zone, workspace, manifest, quick, settings.seed)
    path = context.folder / registry.REGISTRY_FILE
    previous = registry.load(path).models if path.is_file() else []
    entries = {
        entry.name: entry for entry in previous if entry.data_fingerprint == manifest.fingerprint
    }
    for name in names:
        trained = TRAINERS[name](context)
        folder = context.folder / name
        folder.mkdir(parents=True, exist_ok=True)
        results_path = folder / "results.json"
        results_path.write_text(
            json.dumps(_json(trained.results), indent=1, default=str) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        for table, frame in trained.tables.items():
            _write_table(frame, folder / f"{table}.json")
        if name == "silence":
            primary = trained.results["backtests"][-1]
            plots.reliability_png(
                primary["reliability"],
                folder / plots.RELIABILITY_IMAGE,
                f"Reliability on the cutoff of {primary['test_cutoff']}",
            )
        headline = _json(trained.headline)
        run_id = registry.log_run(
            workspace.root / "mlruns",
            f"strata-{zone.value}",
            name,
            params=trained.params,
            headline=headline,
            artefact=results_path,
        )
        entries[name] = registry.entry(
            name=name,
            title=TITLES[name],
            manifest_fingerprint=manifest.fingerprint,
            mapping_sha256=manifest.mapping_sha256,
            source_kind=manifest.source_kind,
            effort="quick" if quick else "full",
            params=_json(trained.params),
            headline=headline,
            run_id=run_id,
            results_file=f"{name}/results.json",
        )
    ordered = [entries[name] for name in MODELS if name in entries]
    result = registry.Registry(zone=zone.value, source_kind=manifest.source_kind, models=ordered)
    registry.write(path, result)
    cards.write_cards(registry.load(path), context.folder / "cards", images=context.folder)
    return result


def promote(settings: Settings) -> list[Path]:
    """Publish the public registry and its model cards into the repository."""
    path = registry.promote(settings)
    published = registry.load(path)
    images = settings.workspace(Zone.PUBLIC).root / "models"
    folder = settings.repo_root / cards.CARDS_DIR
    return [path, *cards.write_cards(published, folder, images=images)]
