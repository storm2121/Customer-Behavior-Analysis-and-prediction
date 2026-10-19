"""The FastAPI application (ARCHITECTURE section 11).

Endpoints read what the pipeline produced in one zone's workspace: the warehouse (a read-only
DuckDB connection per request), the ledger, the model registry and its artefacts, the Markov
simulator's inputs and the review store. Nothing is computed from scratch here except a simulation
for a policy the caller sends. ``/metrics`` exposes Prometheus counters and latency histograms by
route template (never by raw path, which would carry client ids); an optional API key
(``STRATA_API_KEY``) protects every endpoint but ``/health``, ``/metrics`` and the docs.

The API serves the public zone (synthetic data) unless it is started with ``STRATA_ZONE=private``
on the machine that holds the extract.
"""

from __future__ import annotations

import json
import os
import platform
import time
from collections.abc import Awaitable, Callable, Iterator
from typing import Annotated, Any

import duckdb
import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, FastAPI, HTTPException, Query, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Histogram,
    generate_latest,
)
from pydantic import BaseModel, Field

from strata import __version__
from strata.analysis import markov
from strata.config import Settings, Zone
from strata.contracts.specs import WINDOW_END, WINDOW_START
from strata.ingest.load import load_manifest
from strata.ledger.build import PUBLIC_LEDGER, load_ledger
from strata.ledger.model import Ledger
from strata.models.intervals import METRIC_INTERVAL, STABILITY_INTERVAL
from strata.review import db
from strata.warehouse import connect

OPEN_PATHS = {"/health", "/metrics", "/docs", "/openapi.json", "/redoc"}


class PolicyRequest(BaseModel):
    """A what-if: move `share` of branch-only clients younger than `max_age` to `target`."""

    share: float = Field(ge=0.0, le=1.0, default=0.3)
    max_age: int = Field(ge=18, le=200, default=45)
    target: str = Field(default="mixed", pattern="^(mixed|digital_only|none)$")
    horizon: int = Field(ge=1, le=36, default=12)
    samples: int = Field(ge=50, le=5000, default=1000)


class FeedbackRequest(BaseModel):
    label: str = Field(pattern="^(confirmed|dismissed|needs_information)$")
    analyst: str = Field(min_length=1, max_length=128)
    note: str = Field(default="", max_length=2000)


class State:
    """Where one running API reads from."""

    def __init__(self, settings: Settings, zone: Zone) -> None:
        self.settings = settings
        self.zone = zone
        self.workspace = settings.workspace(zone)
        self.store = db.engine(db.database_url(settings, zone))

    def json(self, *parts: str) -> Any:
        path = self.workspace.root.joinpath(*parts)
        if not path.is_file():
            raise HTTPException(status_code=503, detail=f"not available yet: {'/'.join(parts)}")
        return json.loads(path.read_text(encoding="utf-8"))

    def optional(self, *parts: str) -> list[dict[str, Any]]:
        path = self.workspace.root.joinpath(*parts)
        if not path.is_file():
            return []
        rows: list[dict[str, Any]] = json.loads(path.read_text(encoding="utf-8"))
        return rows

    def latest_alerts(self, budget: int | None) -> list[dict[str, Any]]:
        run_id = db.latest_run(self.store)
        return db.alerts_of(self.store, run_id, limit=budget) if run_id is not None else []

    def ledger(self) -> Ledger:
        name = "claims.private.json" if self.workspace.is_private else "claims.synthetic.json"
        path = self.workspace.ledger_dir / name
        if not path.is_file():
            raise HTTPException(status_code=503, detail="ledger not built yet")
        return load_ledger(path)

    def optional_ledger(self) -> Ledger | None:
        try:
            return self.ledger()
        except HTTPException:
            return None


def get_state(request: Request) -> State:
    state: State = request.app.state.strata
    return state


StateDep = Annotated[State, Depends(get_state)]


def get_connection(state: StateDep) -> Iterator[duckdb.DuckDBPyConnection]:
    con = connect(state.workspace.warehouse, read_only=True)
    try:
        yield con
    finally:
        con.close()


ConnectionDep = Annotated[duckdb.DuckDBPyConnection, Depends(get_connection)]

router = APIRouter()


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = json.loads(frame.to_json(orient="records", date_format="iso"))
    return rows


PERIOD = [WINDOW_START.isoformat(), WINDOW_END.isoformat()]
"""The analysis window every claim and every count refers to (the context line of the documents)."""


def _channel_mix_intervals(ledger: Ledger | None, fingerprint: str) -> dict[str, Any] | None:
    """The zone ledger's channel-mix shares with their intervals, so the overview's shares carry
    the ledger's own uncertainty rather than one the dashboard would have to compute. Only from a
    ledger of the data now in the warehouse: a stale one would pair intervals with other counts."""
    if ledger is None or "channel_mix" not in ledger.ids:
        return None
    record = ledger.get("channel_mix")
    if record.data_fingerprint != fingerprint:
        return None
    shares = {
        key.removeprefix("share_"): {"value": value, "ci": record.ci.get(key)}
        for key, value in record.values.items()
        if key.startswith("share_")
    }
    return {"interval": record.interval_label, "n": record.n, "shares": shares}


@router.get("/health")
def health(state: StateDep) -> dict[str, Any]:
    manifest = load_manifest(state.workspace)
    registry_path = state.workspace.root / "models" / "registry.json"
    models = (
        {m["name"]: m["version"] for m in json.loads(registry_path.read_text("utf-8"))["models"]}
        if registry_path.is_file()
        else {}
    )
    return {
        "status": "ok",
        "zone": state.zone.value,
        "versions": {"strata": __version__, "python": platform.python_version()},
        "data": {
            "source_kind": manifest.source_kind,
            "fingerprint": manifest.fingerprint,
            "mapping": manifest.mapping,
        },
        "models": models,
    }


@router.get("/clients/{client_id}/360")
def client_360(client_id: str, state: StateDep, con: ConnectionDep) -> dict[str, Any]:
    profile = con.execute("select * from marts.client_360 where client_id = ?", [client_id]).df()
    if profile.empty:
        raise HTTPException(status_code=404, detail="no such client")
    timeline = con.execute(
        "select month_start, channel_state, n_events from intermediate.int_channel_state "
        "where client_id = ? order by month_start",
        [client_id],
    ).df()
    scored = {row["client_id"]: row for row in state.optional("models", "silence", "scored.json")}
    segments = {
        row["client_id"]: row["segment"]
        for row in state.optional("models", "segments", "assignments.json")
    }
    alerts = [
        alert
        for alert in state.latest_alerts(None)
        if alert["client_id"] == client_id and alert["status"] == "open"
    ]
    return {
        "client_id": client_id,
        "profile": _records(profile)[0],
        "timeline": _records(timeline),
        "segment": segments.get(client_id),
        "silence_risk": scored.get(client_id),
        "open_alerts": alerts,
    }


@router.get("/overview")
def overview(state: StateDep, con: ConnectionDep) -> dict[str, Any]:
    """Headline counts, the channel mix of clients and daily events per channel (heatmap)."""
    profiles = con.execute(
        "select channel_profile, count(*) as clients from marts.client_360 group by 1 order by 1"
    ).df()
    totals = con.execute(
        "select count(*) as clients, coalesce(sum(n_events), 0) as events from marts.client_360"
    ).df()
    daily = con.execute(
        "select day, channel, sum(n_events) as events from marts.daily_counts "
        "group by 1, 2 order by 1, 2"
    ).df()
    manifest = load_manifest(state.workspace)
    return {
        "source_kind": manifest.source_kind,
        "period": PERIOD,
        "clients": int(totals["clients"].iloc[0]),
        "events": int(totals["events"].iloc[0]),
        "channel_mix": _records(profiles),
        "channel_mix_intervals": _channel_mix_intervals(
            state.optional_ledger(), manifest.fingerprint
        ),
        "daily": _records(daily),
    }


@router.get("/segments")
def segment_list(state: StateDep) -> dict[str, Any]:
    """Segments, their stability, and how the map was drawn (UMAP, or principal components at
    quick effort) so the dashboard names the embedding it shows."""
    results = state.json("models", "segments", "results.json")
    out = {key: results[key] for key in ("k", "stability", "segments", "tests")}
    out["stability"] = {**results["stability"], "interval": STABILITY_INTERVAL}
    out["embedding"] = results.get("embedding")
    out["n_clients"] = results.get("n_clients")
    return out


@router.get("/segments/points")
def segment_points(
    state: StateDep, limit: Annotated[int, Query(ge=1, le=50_000)] = 5000
) -> dict[str, Any]:
    """The two-dimensional embedding of the clients, coloured by segment (display only)."""
    rows = state.optional("models", "segments", "assignments.json")
    points = [
        {
            "client_id": row["client_id"],
            "segment": row["segment"],
            "x": row["embed_x"],
            "y": row["embed_y"],
        }
        for row in rows[:limit]
        if "embed_x" in row
    ]
    return {"points": points}


@router.get("/risk/silence/evaluation")
def silence_evaluation(state: StateDep) -> dict[str, Any]:
    """The primary backtest: metrics of every model with intervals, and reliability bins."""
    results = state.json("models", "silence", "results.json")
    primary = results["backtests"][-1]
    out = {
        key: primary[key]
        for key in (
            "train_cutoff",
            "test_cutoff",
            "n_train",
            "n_test",
            "silence_rate_test",
            "metrics",
            "reliability",
        )
    }
    out["interval"] = METRIC_INTERVAL
    return out


@router.get("/risk/silence")
def silence(state: StateDep, top: Annotated[int, Query(ge=1, le=10_000)] = 50) -> dict[str, Any]:
    scored = state.json("models", "silence", "scored.json")
    return {
        "cutoff": scored[0]["cutoff_date"] if scored else None,
        "n_scored": len(scored),
        "clients": scored[:top],
    }


@router.get("/adoption/deciles")
def adoption(state: StateDep) -> dict[str, Any]:
    results = state.json("models", "adoption", "results.json")
    return {
        target: {"rate": item["rate"], "n_clients": item["n_clients"], "deciles": item["deciles"]}
        for target, item in results.items()
    }


@router.get("/alerts")
def alerts(state: StateDep, budget: Annotated[int, Query(ge=1, le=1000)] = 50) -> dict[str, Any]:
    return {"alerts": state.latest_alerts(budget)}


@router.post("/alerts/{alert_id}/feedback", status_code=201)
def feedback(alert_id: int, body: FeedbackRequest, state: StateDep) -> dict[str, Any]:
    try:
        feedback_id = db.add_feedback(state.store, alert_id, body.label, body.analyst, body.note)
    except KeyError:
        raise HTTPException(status_code=404, detail="no such alert") from None
    return {"feedback_id": feedback_id, "alert_id": alert_id, "label": body.label}


@router.get("/calendar/effects")
def calendar_effects(state: StateDep) -> dict[str, Any]:
    ledger = state.ledger()
    out: dict[str, Any] = {}
    for claim_id in ("calendar_effects", "calendar_bayes"):
        if claim_id in ledger.ids:
            record = ledger.get(claim_id)
            out[claim_id] = {
                "interval": record.interval_label,
                "effects": {
                    key: {"value": value, "ci": record.ci.get(key)}
                    for key, value in record.values.items()
                },
                "caveats": record.caveats,
            }
    if not out:
        raise HTTPException(status_code=503, detail="calendar claims not built yet")
    return out


@router.post("/markov/simulate")
def simulate(policy: PolicyRequest, state: StateDep) -> dict[str, Any]:
    inputs = state.json("analysis", "markov_simulator.json")
    by_band = {
        band: np.asarray(counts, dtype=np.float64)
        for band, counts in inputs["start_counts_by_age_band"].items()
    }
    eligible = [band["band"] for band in inputs["age_bands"] if band["upper"] <= policy.max_age]
    start = markov.banded_start(by_band, eligible, 0.0)
    moved = markov.banded_start(by_band, eligible, policy.share, policy.target)
    counts = np.asarray(inputs["transition_counts"], dtype=np.float64)
    samples = markov.posterior_samples(counts, n_samples=policy.samples, seed=state.settings.seed)
    rates = inputs["branch_rates"]
    baseline = markov.bands(markov.simulate_paths(samples, start, rates, policy.horizon))
    result = markov.bands(markov.simulate_paths(samples, moved, rates, policy.horizon))
    return {
        "policy": policy.model_dump(),
        "interval": markov.BAND_INTERVAL,
        "eligible_bands": eligible,
        "months": result.months,
        "policy_bands": result.quantiles,
        "baseline_bands": baseline.quantiles,
    }


@router.get("/ledger/claims")
def claims(state: StateDep) -> dict[str, Any]:
    path = state.settings.repo_root / PUBLIC_LEDGER
    if not path.is_file():
        raise HTTPException(status_code=503, detail="no public ledger")
    public = load_ledger(path)
    # Everything a claim card shows comes from the public ledger itself: the method, the caveats,
    # the robustness explanation, the period and the source travel with every number.
    return {
        "data_fingerprint": public.data_fingerprint,
        "source_kind": public.source_kind,
        "claims": [
            {
                "id": record.id,
                "title": record.title,
                "question": record.question,
                "method": record.method,
                "values": record.values,
                "units": {key: unit.value for key, unit in record.units.items()},
                "ci": record.ci,
                "interval": record.interval_label,
                "n": record.n,
                "n_events": record.n_events,
                "period": PERIOD,
                "source_kind": record.source_kind,
                "caveats": record.caveats,
                "verdict": record.robustness.verdict if record.robustness else None,
                "mechanism": record.robustness.mechanism if record.robustness else None,
            }
            for record in public.claims
        ],
        "verdicts": [verdict.model_dump(mode="json") for verdict in public.verdicts],
    }


def create_app(settings: Settings | None = None, zone: Zone | None = None) -> FastAPI:
    """Build the application for ``zone`` (default: the settings' zone)."""
    settings = settings or Settings()
    app = FastAPI(
        title="Strata API",
        version=__version__,
        description="Risk scores, segments, alerts, the policy simulator and the public claims.",
    )
    app.state.strata = State(settings, zone or settings.zone)
    origins = os.environ.get(
        "STRATA_CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173"
    ).split(",")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[origin.strip() for origin in origins if origin.strip()],
        allow_methods=["GET", "POST"],
        allow_headers=["content-type", "x-api-key"],
    )
    metrics = CollectorRegistry()
    requests = Counter(
        "strata_requests", "Requests by route and status", ["route", "status"], registry=metrics
    )
    latency = Histogram("strata_request_seconds", "Request latency", ["route"], registry=metrics)
    api_key = os.environ.get("STRATA_API_KEY")

    @app.middleware("http")
    async def observe(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        started = time.perf_counter()
        protected = request.url.path not in OPEN_PATHS
        if api_key and protected and request.headers.get("x-api-key") != api_key:
            response: Response = JSONResponse(
                {"detail": "missing or wrong API key"}, status_code=401
            )
        else:
            response = await call_next(request)
        name = getattr(request.scope.get("route"), "path", "unmatched")
        requests.labels(route=name, status=str(response.status_code)).inc()
        latency.labels(route=name).observe(time.perf_counter() - started)
        return response

    @app.get("/metrics", include_in_schema=False)
    def prometheus() -> Response:
        return Response(generate_latest(metrics), media_type=CONTENT_TYPE_LATEST)

    app.include_router(router)
    return app


def main() -> None:  # pragma: no cover - the container's entry point
    import uvicorn  # noqa: PLC0415

    uvicorn.run(
        "strata.api.app:create_app",
        factory=True,
        host=os.environ.get("STRATA_API_HOST", "0.0.0.0"),  # noqa: S104 - inside a container
        port=int(os.environ.get("STRATA_API_PORT", "8000")),
    )
