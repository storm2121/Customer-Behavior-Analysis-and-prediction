"""The initial review schema: runs, alerts, feedback and annotations.

Revision ID: 0001
Revises:
Create Date: 2026-10-18
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "runs",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("zone", sa.String(16), nullable=False),
        sa.Column("source_kind", sa.String(16), nullable=False),
        sa.Column("data_fingerprint", sa.String(64), nullable=False),
        sa.Column("rules_digest", sa.String(64), nullable=False),
        sa.Column("budget", sa.Integer(), nullable=False),
        sa.Column("n_clients", sa.Integer(), nullable=False),
    )
    op.create_table(
        "alerts",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column(
            "run_id", sa.Integer(), sa.ForeignKey("runs.id", ondelete="CASCADE"), nullable=False
        ),
        sa.Column("client_id", sa.String(128), nullable=False),
        sa.Column("rank", sa.Integer(), nullable=False),
        sa.Column("score", sa.Float(), nullable=False),
        sa.Column("rules", sa.JSON(), nullable=False),
        sa.Column("reasons", sa.JSON(), nullable=False),
        sa.Column("status", sa.String(32), nullable=False),
    )
    op.create_index("ix_alerts_run_id", "alerts", ["run_id"])
    op.create_index("ix_alerts_client_id", "alerts", ["client_id"])
    op.create_table(
        "feedback",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column(
            "alert_id", sa.Integer(), sa.ForeignKey("alerts.id", ondelete="CASCADE"), nullable=False
        ),
        sa.Column("label", sa.String(32), nullable=False),
        sa.Column("analyst", sa.String(128), nullable=False),
        sa.Column("note", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_feedback_alert_id", "feedback", ["alert_id"])
    op.create_table(
        "annotations",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column(
            "alert_id", sa.Integer(), sa.ForeignKey("alerts.id", ondelete="CASCADE"), nullable=False
        ),
        sa.Column("author", sa.String(128), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_annotations_alert_id", "annotations", ["alert_id"])


def downgrade() -> None:
    op.drop_table("annotations")
    op.drop_table("feedback")
    op.drop_table("alerts")
    op.drop_table("runs")
