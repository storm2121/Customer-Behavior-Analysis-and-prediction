"""Smoke test of a running stack: every API endpoint answers on synthetic data.

    python docker/smoke.py [api_url] [dashboard_url]

With a dashboard URL, the dashboard's page and the API behind its same-origin proxy must answer
too. Exits non-zero on the first check that does not answer as expected.
"""

from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.request

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"
DASHBOARD = sys.argv[2] if len(sys.argv) > 2 else None
if not all(url.startswith(("http://", "https://")) for url in (BASE, DASHBOARD or BASE)):
    sys.exit("the URLs must be http or https")


def call(method: str, path: str, body: dict[str, object] | None = None) -> tuple[int, object]:
    data = json.dumps(body).encode() if body is not None else None
    request = urllib.request.Request(  # noqa: S310 - the scheme is checked above
        BASE + path, data=data, method=method, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:  # noqa: S310 - checked above
            raw = response.read().decode()
            status = response.status
    except urllib.error.HTTPError as error:
        return error.code, error.read().decode()
    try:
        return status, json.loads(raw)
    except json.JSONDecodeError:
        return status, raw


def wait_for_health(timeout: float = 600.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            if call("GET", "/health")[0] == 200:
                return
        except (urllib.error.URLError, ConnectionError):
            pass
        time.sleep(3)
    sys.exit("the API did not become healthy")


def main() -> None:
    wait_for_health()
    status, risk = call("GET", "/risk/silence?top=3")
    if status != 200 or not isinstance(risk, dict):
        sys.exit(f"/risk/silence answered {status}")
    client = risk["clients"][0]["client_id"]
    status, alerts = call("GET", "/alerts?budget=5")
    if status != 200 or not isinstance(alerts, dict):
        sys.exit(f"/alerts answered {status}")
    alert = alerts["alerts"][0]["id"]
    checks = [
        ("GET", "/health", None, 200),
        ("GET", f"/clients/{client}/360", None, 200),
        ("GET", "/overview", None, 200),
        ("GET", "/segments", None, 200),
        ("GET", "/segments/points?limit=100", None, 200),
        ("GET", "/risk/silence?top=10", None, 200),
        ("GET", "/risk/silence/evaluation", None, 200),
        ("GET", "/adoption/deciles", None, 200),
        ("GET", "/alerts?budget=10", None, 200),
        ("POST", f"/alerts/{alert}/feedback", {"label": "dismissed", "analyst": "smoke"}, 201),
        ("GET", "/calendar/effects", None, 200),
        ("POST", "/markov/simulate", {"share": 0.3, "max_age": 45, "samples": 200}, 200),
        ("GET", "/ledger/claims", None, 200),
        ("GET", "/metrics", None, 200),
        ("GET", "/docs", None, 200),
    ]
    for method, path, body, expected in checks:
        status, _ = call(method, path, body)
        sys.stdout.write(f"{status} {method} {path}\n")
        if status != expected:
            sys.exit(f"expected {expected} from {method} {path}")
    if DASHBOARD is not None:
        dashboard_checks(DASHBOARD.rstrip("/"))
    sys.stdout.write("every endpoint answered\n")


def dashboard_checks(dashboard: str) -> None:
    """The dashboard's page, and the API through the dashboard's same-origin proxy."""
    for path in ("/", "/api/health"):
        try:
            with urllib.request.urlopen(dashboard + path, timeout=60) as response:  # noqa: S310
                status = response.status
        except urllib.error.HTTPError as error:
            status = error.code
        sys.stdout.write(f"{status} GET {dashboard}{path}\n")
        if status != 200:
            sys.exit(f"expected 200 from {dashboard}{path}")


if __name__ == "__main__":
    main()
