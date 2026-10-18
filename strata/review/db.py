"""Connecting to the review store, migrating it, and the operations the CLI and the API share.

Where the store lives:

- public zone: ``STRATA_PG_URL`` when set (the PostgreSQL service of Docker Compose), otherwise a
  SQLite file in the workspace (``.strata/review/review.sqlite``);
- private zone: always a SQLite file under ``STRATA_PRIVATE_DIR``. Alerts name real clients, so they
  never go to a server whose storage the privacy rules cannot see.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

from alembic import command
from alembic.config import Config
from sqlalchemy import Engine, create_engine, func, select
from sqlalchemy.orm import Session

from strata.config import Settings, Zone
from strata.review.models import LABELS, Alert, Annotation, Feedback, Run

MIGRATIONS = Path(__file__).with_name("migrations")


def database_url(settings: Settings, zone: Zone) -> str:
    """The SQLAlchemy URL of the zone's review store."""
    workspace = settings.workspace(zone)
    if zone is Zone.PUBLIC and settings.pg_url:
        return settings.pg_url
    path = workspace.root / "review" / "review.sqlite"
    if workspace.is_private:
        settings.assert_private_output(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{path.as_posix()}"


def migrate(url: str, revision: str = "head") -> None:
    """Bring the schema to ``revision`` with Alembic."""
    config = Config()
    config.set_main_option("script_location", MIGRATIONS.as_posix())
    config.set_main_option("sqlalchemy.url", url.replace("%", "%%"))
    command.upgrade(config, revision)


def downgrade(url: str, revision: str = "base") -> None:
    config = Config()
    config.set_main_option("script_location", MIGRATIONS.as_posix())
    config.set_main_option("sqlalchemy.url", url.replace("%", "%%"))
    command.downgrade(config, revision)


def engine(url: str) -> Engine:
    """An engine on a migrated store."""
    migrate(url)
    return create_engine(url, future=True)


def store_run(
    connection: Engine,
    *,
    zone: str,
    source_kind: str,
    data_fingerprint: str,
    rules_digest: str,
    n_clients: int,
    alerts: Iterable[dict[str, Any]],
    budget: int,
) -> int:
    """Store one run and its budgeted queue; returns the run id."""
    with Session(connection) as session, session.begin():
        run = Run(
            zone=zone,
            source_kind=source_kind,
            data_fingerprint=data_fingerprint,
            rules_digest=rules_digest,
            budget=budget,
            n_clients=n_clients,
        )
        run.alerts = [
            Alert(
                client_id=str(item["client_id"]),
                rank=int(item["rank"]),
                score=float(item["score"]),
                rules=list(item["rules"]),
                reasons=list(item["reasons"]),
                status="open",
            )
            for item in alerts
        ]
        session.add(run)
        session.flush()
        return int(run.id)


def latest_run(connection: Engine) -> int | None:
    with Session(connection) as session:
        found = session.scalar(select(func.max(Run.id)))
        return int(found) if found is not None else None


def alerts_of(connection: Engine, run_id: int, limit: int | None = None) -> list[dict[str, Any]]:
    """The queue of a run, in rank order, with each alert's latest label."""
    with Session(connection) as session:
        query = select(Alert).where(Alert.run_id == run_id).order_by(Alert.rank)
        if limit is not None:
            query = query.limit(limit)
        rows = session.scalars(query).all()
        return [
            {
                "id": alert.id,
                "client_id": alert.client_id,
                "rank": alert.rank,
                "score": alert.score,
                "rules": alert.rules,
                "reasons": alert.reasons,
                "status": alert.status,
                "label": _latest_label(alert),
            }
            for alert in rows
        ]


def _latest_label(alert: Alert) -> str | None:
    if not alert.feedback:
        return None
    return max(alert.feedback, key=lambda item: (item.created_at, item.id)).label


def add_feedback(
    connection: Engine, alert_id: int, label: str, analyst: str, note: str = ""
) -> int:
    """Record an analyst's label; the alert's status follows the latest label."""
    if label not in LABELS:
        msg = f"label must be one of {', '.join(LABELS)}"
        raise ValueError(msg)
    with Session(connection) as session, session.begin():
        alert = session.get(Alert, alert_id)
        if alert is None:
            msg = f"no alert {alert_id}"
            raise KeyError(msg)
        item = Feedback(alert_id=alert_id, label=label, analyst=analyst, note=note)
        session.add(item)
        alert.status = "reviewed"
        session.flush()
        return int(item.id)


def annotate(connection: Engine, alert_id: int, author: str, text: str) -> int:
    with Session(connection) as session, session.begin():
        if session.get(Alert, alert_id) is None:
            msg = f"no alert {alert_id}"
            raise KeyError(msg)
        item = Annotation(alert_id=alert_id, author=author, text=text)
        session.add(item)
        session.flush()
        return int(item.id)


def precision_from_feedback(connection: Engine, run_id: int, k: int) -> dict[str, float | int]:
    """Precision at ``k`` against analyst labels: confirmed over labelled, among the top ``k``."""
    labels = [item["label"] for item in alerts_of(connection, run_id, limit=k)]
    labelled = [label for label in labels if label in {"confirmed", "dismissed"}]
    confirmed = sum(1 for label in labelled if label == "confirmed")
    return {
        "k": k,
        "labelled": len(labelled),
        "confirmed": confirmed,
        "precision": confirmed / len(labelled) if labelled else float("nan"),
    }
