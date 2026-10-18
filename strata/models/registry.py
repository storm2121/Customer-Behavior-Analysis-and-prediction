"""Model registry: MLflow runs in the zone's workspace, and the public ``models/registry.json``.

Every training run is logged to an MLflow file store in the zone's workspace (``.strata/mlruns`` in
the public zone, ``<STRATA_PRIVATE_DIR>/mlruns`` in the private one): parameters, headline metrics
and the full results as an artefact. The zone's registry (``models/registry.json`` in the
workspace) lists the latest run of every model. ``strata promote`` copies the public registry into
the repository, and only a registry of models trained on synthetic data can be promoted.
"""

from __future__ import annotations

import json
import logging
import math
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

from strata import __version__
from strata.config import PrivacyBoundaryError, Settings, Zone

PUBLIC_REGISTRY = Path("models/registry.json")
REGISTRY_FILE = "registry.json"


class ModelEntry(BaseModel):
    """One registered model."""

    model_config = ConfigDict(frozen=True)

    name: str
    title: str
    version: str
    trained_at: datetime
    source_kind: str
    data_fingerprint: str
    mapping_sha256: str | None
    strata_version: str
    effort: str
    params: dict[str, Any]
    headline: dict[str, Any]
    mlflow_run_id: str | None
    results_file: str


class Registry(BaseModel):
    model_config = ConfigDict(frozen=True)

    zone: str
    source_kind: str
    models: list[ModelEntry]

    def get(self, name: str) -> ModelEntry:
        for entry in self.models:
            if entry.name == name:
                return entry
        msg = f"no model {name!r} in the registry"
        raise KeyError(msg)


def _flat_metrics(headline: dict[str, Any], prefix: str = "") -> dict[str, float]:
    out: dict[str, float] = {}
    for key, value in headline.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict) and "value" in value:
            out[name] = float(value["value"])
        elif isinstance(value, dict):
            out.update(_flat_metrics(value, f"{name}."))
        elif isinstance(value, int | float) and not isinstance(value, bool):
            out[name] = float(value)
    return out


def log_run(
    mlruns: Path,
    experiment: str,
    name: str,
    *,
    params: dict[str, Any],
    headline: dict[str, Any],
    artefact: Path,
) -> str | None:
    """Log one training run to the MLflow file store; returns the run id."""
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    # ARCHITECTURE section 10 keeps runs in a plain file store in the workspace; MLflow 3 asks for
    # an explicit opt-in to it.
    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    import mlflow  # noqa: PLC0415 - heavy import, only when training

    logging.getLogger("mlflow").setLevel(logging.ERROR)
    mlruns.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(mlruns.resolve().as_uri())
    mlflow.set_experiment(experiment)
    with mlflow.start_run(run_name=name) as run:
        mlflow.log_params({key: str(value) for key, value in params.items()})
        flat = {key.replace("@", "_at_"): value for key, value in _flat_metrics(headline).items()}
        mlflow.log_metrics({k: v for k, v in flat.items() if math.isfinite(v)})
        mlflow.log_artifact(str(artefact))
        return str(run.info.run_id)


def entry(
    *,
    name: str,
    title: str,
    manifest_fingerprint: str,
    mapping_sha256: str | None,
    source_kind: str,
    effort: str,
    params: dict[str, Any],
    headline: dict[str, Any],
    run_id: str | None,
    results_file: str,
) -> ModelEntry:
    trained_at = datetime.now(UTC).replace(microsecond=0)
    return ModelEntry(
        name=name,
        title=title,
        version=f"{__version__}+{manifest_fingerprint[:8]}.{trained_at:%Y%m%d%H%M%S}",
        trained_at=trained_at,
        source_kind=source_kind,
        data_fingerprint=manifest_fingerprint,
        mapping_sha256=mapping_sha256,
        strata_version=__version__,
        effort=effort,
        params=params,
        headline=headline,
        mlflow_run_id=run_id,
        results_file=results_file,
    )


def write(path: Path, registry: Registry) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = registry.model_dump(mode="json")
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )
    return path


def load(path: Path) -> Registry:
    if not path.is_file():
        msg = f"No registry at {path.name}: run `strata train` first."
        raise FileNotFoundError(msg)
    return Registry.model_validate_json(path.read_text(encoding="utf-8"))


def promote(settings: Settings) -> Path:
    """Copy the public zone's registry into the repository (synthetic-trained models only)."""
    source = settings.workspace(Zone.PUBLIC).root / "models" / REGISTRY_FILE
    registry = load(source)
    if registry.source_kind != "synthetic" or any(
        entry.source_kind != "synthetic" for entry in registry.models
    ):
        msg = "Only models trained on synthetic data can be promoted into the repository."
        raise PrivacyBoundaryError(msg)
    # MLflow run ids point into a local store and look like opaque identifiers; they stay local.
    public = registry.model_copy(
        update={
            "models": [
                entry.model_copy(update={"mlflow_run_id": None}) for entry in registry.models
            ]
        }
    )
    return write(settings.repo_root / PUBLIC_REGISTRY, public)
