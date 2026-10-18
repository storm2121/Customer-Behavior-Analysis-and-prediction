"""Tables of the review store (migrations in ``strata/review/migrations``)."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from sqlalchemy import JSON, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

LABELS = ("confirmed", "dismissed", "needs_information")
"""What an analyst may say about an alert. ``confirmed`` means worth escalating, not guilt."""


def _now() -> datetime:
    return datetime.now(UTC)


class Base(DeclarativeBase):
    """Declarative base of the review store."""


class Run(Base):
    """One execution of the monitoring layer."""

    __tablename__ = "runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    zone: Mapped[str] = mapped_column(String(16))
    source_kind: Mapped[str] = mapped_column(String(16))
    data_fingerprint: Mapped[str] = mapped_column(String(64))
    rules_digest: Mapped[str] = mapped_column(String(64))
    budget: Mapped[int] = mapped_column(Integer)
    n_clients: Mapped[int] = mapped_column(Integer)
    alerts: Mapped[list[Alert]] = relationship(back_populates="run", cascade="all, delete-orphan")


class Alert(Base):
    """One client in a run's queue, with the reasons a reviewer needs."""

    __tablename__ = "alerts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    run_id: Mapped[int] = mapped_column(ForeignKey("runs.id", ondelete="CASCADE"), index=True)
    client_id: Mapped[str] = mapped_column(String(128), index=True)
    rank: Mapped[int] = mapped_column(Integer)
    score: Mapped[float] = mapped_column(Float)
    rules: Mapped[list[Any]] = mapped_column(JSON)
    reasons: Mapped[list[Any]] = mapped_column(JSON)
    status: Mapped[str] = mapped_column(String(32), default="open")
    run: Mapped[Run] = relationship(back_populates="alerts")
    feedback: Mapped[list[Feedback]] = relationship(
        back_populates="alert", cascade="all, delete-orphan"
    )
    annotations: Mapped[list[Annotation]] = relationship(
        back_populates="alert", cascade="all, delete-orphan"
    )


class Feedback(Base):
    """An analyst's label for an alert; the latest label counts."""

    __tablename__ = "feedback"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    alert_id: Mapped[int] = mapped_column(ForeignKey("alerts.id", ondelete="CASCADE"), index=True)
    label: Mapped[str] = mapped_column(String(32))
    analyst: Mapped[str] = mapped_column(String(128))
    note: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    alert: Mapped[Alert] = relationship(back_populates="feedback")


class Annotation(Base):
    """A free-text note on an alert."""

    __tablename__ = "annotations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    alert_id: Mapped[int] = mapped_column(ForeignKey("alerts.id", ondelete="CASCADE"), index=True)
    author: Mapped[str] = mapped_column(String(128))
    text: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    alert: Mapped[Alert] = relationship(back_populates="annotations")
