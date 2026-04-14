"""SQLAlchemy 2.x ORM models for persistence."""

from datetime import datetime
from typing import Any

from sqlalchemy import JSON, DateTime, Float, Integer, String, Text, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """Base class for all ORM models."""


class AnalysisRun(Base):
    """A single analysis run comparing one or more repos."""

    __tablename__ = "analysis_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    repo_urls: Mapped[dict[str, Any]] = mapped_column(JSON, default=list)
    status: Mapped[str] = mapped_column(String(32), default="pending")
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    completed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)


class EvidenceRecord(Base):
    """Persisted evidence pack for a repo."""

    __tablename__ = "evidence_records"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(String(64), index=True)
    repo_name: Mapped[str] = mapped_column(String(256))
    evidence_json: Mapped[dict[str, Any]] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class ScoreRecord(Base):
    """Persisted scorecard for a repo."""

    __tablename__ = "score_records"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(String(64), index=True)
    repo_name: Mapped[str] = mapped_column(String(256))
    scores_json: Mapped[dict[str, Any]] = mapped_column(JSON)
    weighted_total: Mapped[float] = mapped_column(Float, default=0.0)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class DebateRecordDB(Base):
    """Persisted debate record."""

    __tablename__ = "debate_records"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(String(64), index=True)
    debate_json: Mapped[dict[str, Any]] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class ReportRecord(Base):
    """Persisted final synthesis report."""

    __tablename__ = "report_records"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    report_json: Mapped[dict[str, Any]] = mapped_column(JSON)
    report_text: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class ScanRecord(Base):
    """Persisted scanner run result."""

    __tablename__ = "scan_records"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    scan_id: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    tickers: Mapped[dict[str, Any]] = mapped_column(JSON, default=list)
    result_json: Mapped[dict[str, Any]] = mapped_column(JSON)
    top_score: Mapped[float] = mapped_column(Float, default=0.0)
    opportunity_count: Mapped[int] = mapped_column(Integer, default=0)
    status: Mapped[str] = mapped_column(String(32), default="complete")
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
