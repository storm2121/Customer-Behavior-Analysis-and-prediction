"""The HTTP API on a synthetic workspace: every endpoint, metrics, the API key and errors."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from strata.analysis.run import run_module
from strata.api.app import create_app
from strata.config import Settings, Zone
from strata.ledger.build import build_ledger
from strata.models.anomaly.run import run as run_anomaly
from strata.models.train import train


@pytest.fixture(scope="module")
def api_settings(built_warehouse: Settings) -> Settings:
    """The synthetic workspace with everything the API serves: claims, models, simulator, alerts."""
    build_ledger(built_warehouse, Zone.PUBLIC, only=["calendar_effects"], quick=True)
    run_module(built_warehouse, Zone.PUBLIC, "markov", quick=True)
    train(built_warehouse, Zone.PUBLIC, ["silence", "adoption", "segments"], quick=True)
    run_anomaly(built_warehouse, Zone.PUBLIC, budget=20)
    return built_warehouse


@pytest.fixture(scope="module")
def client(api_settings: Settings) -> TestClient:
    return TestClient(create_app(api_settings, Zone.PUBLIC))


def test_health_reports_versions_data_and_models(client: TestClient) -> None:
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["data"]["source_kind"] == "synthetic"
    assert len(body["data"]["fingerprint"]) == 64
    assert {"silence", "adoption", "segments"} <= set(body["models"])


def test_client_360_combines_profile_risk_segment_and_alerts(client: TestClient) -> None:
    top = client.get("/risk/silence", params={"top": 1}).json()["clients"][0]
    body = client.get(f"/clients/{top['client_id']}/360").json()
    assert body["profile"]["client_id"] == top["client_id"]
    assert body["timeline"]
    assert body["silence_risk"]["probability"] == pytest.approx(top["probability"])
    assert body["segment"] is not None
    assert isinstance(body["open_alerts"], list)
    assert client.get("/clients/nobody/360").status_code == 404


def test_risk_segments_adoption_and_calendar(client: TestClient) -> None:
    risk = client.get("/risk/silence", params={"top": 5}).json()
    probabilities = [row["probability"] for row in risk["clients"]]
    assert probabilities == sorted(probabilities, reverse=True)
    assert len(probabilities) == 5
    segments = client.get("/segments").json()
    assert segments["k"] == len(segments["segments"])
    deciles = client.get("/adoption/deciles").json()
    assert {"target_has_online", "target_active_online"} == set(deciles)
    assert len(deciles["target_has_online"]["deciles"]) == 10
    calendar = client.get("/calendar/effects").json()
    effects = calendar["calendar_effects"]["effects"]
    assert all(len(item["ci"]) == 2 for item in effects.values() if item["ci"])
    assert client.get("/risk/silence", params={"top": 0}).status_code == 422


def test_alerts_and_feedback_round_trip(client: TestClient) -> None:
    alerts = client.get("/alerts", params={"budget": 5}).json()["alerts"]
    assert [alert["rank"] for alert in alerts] == [1, 2, 3, 4, 5]
    assert alerts[0]["rules"]
    first = alerts[0]["id"]
    created = client.post(
        f"/alerts/{first}/feedback", json={"label": "dismissed", "analyst": "ana", "note": "known"}
    )
    assert created.status_code == 201
    labelled = client.get("/alerts", params={"budget": 1}).json()["alerts"][0]
    assert labelled["label"] == "dismissed"
    assert (
        client.post(
            "/alerts/999999/feedback", json={"label": "confirmed", "analyst": "a"}
        ).status_code
        == 404
    )
    assert (
        client.post(
            f"/alerts/{first}/feedback", json={"label": "guilty", "analyst": "a"}
        ).status_code
        == 422
    )


def test_the_simulator_returns_bands_for_a_policy(client: TestClient) -> None:
    body = client.post(
        "/markov/simulate", json={"share": 0.3, "max_age": 45, "horizon": 6, "samples": 200}
    ).json()
    assert body["eligible_bands"] == ["<25", "25-34", "35-44"]
    assert body["months"] == list(range(7))
    for bands in (body["policy_bands"], body["baseline_bands"]):
        low, mid, high = (bands["digital_share"][level] for level in ("p05", "p50", "p95"))
        assert all(a <= b <= c for a, b, c in zip(low, mid, high, strict=True))
    gain = (
        body["policy_bands"]["digital_share"]["p50"][0]
        - body["baseline_bands"]["digital_share"]["p50"][0]
    )
    assert gain > 0
    none = client.post("/markov/simulate", json={"share": 0.0, "horizon": 3, "samples": 50}).json()
    assert none["policy_bands"] == none["baseline_bands"]
    assert client.post("/markov/simulate", json={"share": 2}).status_code == 422
    assert client.post("/markov/simulate", json={"target": "branch_only"}).status_code == 422


def test_the_public_claims_are_served(client: TestClient) -> None:
    body = client.get("/ledger/claims").json()
    ids = {claim["id"] for claim in body["claims"]}
    assert "channel_mix" in ids
    for claim in body["claims"]:
        assert claim["question"]
        assert set(claim["units"]) <= set(claim["values"])
    assert {verdict["id"] for verdict in body["verdicts"]} == {"silence_by_channel"}
    assert all("values" not in verdict for verdict in body["verdicts"])


def test_metrics_count_requests_by_route_template(client: TestClient) -> None:
    client.get("/clients/nobody/360")
    text = client.get("/metrics").text
    assert 'strata_requests_total{route="/clients/{client_id}/360",status="404"}' in text
    assert "nobody" not in text
    assert "strata_request_seconds_bucket" in text


def test_the_api_key_protects_everything_but_health(
    api_settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("STRATA_API_KEY", "k3y")
    guarded = TestClient(create_app(api_settings, Zone.PUBLIC))
    assert guarded.get("/health").status_code == 200
    assert guarded.get("/segments").status_code == 401
    assert guarded.get("/segments", headers={"x-api-key": "k3y"}).status_code == 200
    assert guarded.get("/openapi.json").status_code == 200


def test_missing_artefacts_answer_service_unavailable(built_warehouse: Settings, tmp_path) -> None:  # type: ignore[no-untyped-def]
    empty = Settings(  # type: ignore[call-arg]
        _env_file=None, db=built_warehouse.db, work_dir=tmp_path / "empty"
    )
    (tmp_path / "empty").mkdir()
    (tmp_path / "empty" / "manifest.json").write_text(
        (built_warehouse.work_dir / "manifest.json").read_text(encoding="utf-8"), encoding="utf-8"
    )
    bare = TestClient(create_app(empty, Zone.PUBLIC))
    assert bare.get("/segments").status_code == 503
    assert bare.post("/markov/simulate", json={}).status_code == 503
    assert bare.get("/calendar/effects").status_code == 503
    assert bare.get("/alerts").json() == {"alerts": []}


def test_overview_points_and_risk_evaluation_feed_the_dashboard(client: TestClient) -> None:
    overview = client.get("/overview").json()
    assert overview["source_kind"] == "synthetic"
    assert sum(row["clients"] for row in overview["channel_mix"]) == overview["clients"]
    assert {row["channel"] for row in overview["daily"]} == {"card", "online", "branch"}
    points = client.get("/segments/points", params={"limit": 50}).json()["points"]
    assert len(points) == 50
    assert {"client_id", "segment", "x", "y"} <= set(points[0])
    evaluation = client.get("/risk/silence/evaluation").json()
    assert "lightgbm_platt" in evaluation["metrics"]
    assert evaluation["reliability"]["lightgbm_platt"]


def test_cors_allows_the_dashboard_origin(client: TestClient) -> None:
    answer = client.options(
        "/health",
        headers={"Origin": "http://localhost:5173", "Access-Control-Request-Method": "GET"},
    )
    assert answer.headers.get("access-control-allow-origin") == "http://localhost:5173"
