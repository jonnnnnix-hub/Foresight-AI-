"""init all tables

Revision ID: 001
Revises: None
Create Date: 2026-04-11
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "analysis_runs",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("run_id", sa.String(64), nullable=False),
        sa.Column("repo_urls", sa.JSON(), nullable=True),
        sa.Column("status", sa.String(32), nullable=False, server_default="pending"),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now(), nullable=False),
        sa.Column("completed_at", sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("run_id"),
    )
    op.create_index("ix_analysis_runs_run_id", "analysis_runs", ["run_id"])

    op.create_table(
        "evidence_records",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("run_id", sa.String(64), nullable=False),
        sa.Column("repo_name", sa.String(256), nullable=False),
        sa.Column("evidence_json", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_evidence_records_run_id", "evidence_records", ["run_id"])

    op.create_table(
        "score_records",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("run_id", sa.String(64), nullable=False),
        sa.Column("repo_name", sa.String(256), nullable=False),
        sa.Column("scores_json", sa.JSON(), nullable=False),
        sa.Column("weighted_total", sa.Float(), nullable=False, server_default="0"),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_score_records_run_id", "score_records", ["run_id"])

    op.create_table(
        "debate_records",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("run_id", sa.String(64), nullable=False),
        sa.Column("debate_json", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_debate_records_run_id", "debate_records", ["run_id"])

    op.create_table(
        "report_records",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("run_id", sa.String(64), nullable=False),
        sa.Column("report_json", sa.JSON(), nullable=False),
        sa.Column("report_text", sa.Text(), nullable=False, server_default=""),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("run_id"),
    )
    op.create_index("ix_report_records_run_id", "report_records", ["run_id"])

    op.create_table(
        "scan_records",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("scan_id", sa.String(64), nullable=False),
        sa.Column("tickers", sa.JSON(), nullable=True),
        sa.Column("result_json", sa.JSON(), nullable=False),
        sa.Column("top_score", sa.Float(), nullable=False, server_default="0"),
        sa.Column("opportunity_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("status", sa.String(32), nullable=False, server_default="complete"),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("scan_id"),
    )
    op.create_index("ix_scan_records_scan_id", "scan_records", ["scan_id"])


def downgrade() -> None:
    op.drop_table("scan_records")
    op.drop_table("report_records")
    op.drop_table("debate_records")
    op.drop_table("score_records")
    op.drop_table("evidence_records")
    op.drop_table("analysis_runs")
