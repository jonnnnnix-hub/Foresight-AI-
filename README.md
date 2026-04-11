# FlowEdge Repo Intelligence Engine

Production-grade repo analysis platform that ingests GitHub repositories, extracts structured evidence, runs specialist agent reviews, conducts adversarial debate, scores repos against a fixed rubric, and produces a final synthesis report.

## Target Use Case

Compare trading system repos (vectorbt, NautilusTrader, Freqtrade, Qlib, FinRL, hftbacktest) optimized for short-horizon scalp trading workflows.

## Architecture

```
Ingestion → Evidence Extraction → Specialist Agents → Debate → Scoring → Synthesis Report
```

**Layers:**
1. **Ingestion** — Clone and validate GitHub repos
2. **Extraction** — Scan files, classify evidence, build evidence packs
3. **Specialists** — 8 analyst agents evaluate different dimensions
4. **Debate** — Adversarial rounds where agents challenge each other
5. **Scoring** — Weighted rubric across 6 dimensions
6. **Synthesis** — Final report with rankings, borrow/avoid/replace matrix, and merged architecture

## Stack

- Python 3.12, LangGraph, Pydantic v2
- FastAPI + Typer + SQLAlchemy 2.x + Alembic
- PostgreSQL, structlog, pytest, ruff, mypy

## Quick Start

```bash
# Install dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Lint
ruff check .

# Type check
mypy src

# CLI
flowedge analyze https://github.com/polakowo/vectorbt

# API
uvicorn flowedge.api.app:app --reload
```

## Scoring Dimensions

| Dimension | Weight |
|-----------|--------|
| Research Power | 0.15 |
| Execution Realism | 0.20 |
| ML Readiness | 0.15 |
| Productization Value | 0.15 |
| Reliability | 0.15 |
| Scalp Fit | 0.20 |

## Project Structure

```
src/flowedge/
├── agents/       # Specialist analyst agents
├── api/          # FastAPI routes
├── cli/          # Typer CLI
├── config/       # Settings and logging
├── db/           # SQLAlchemy models and sessions
├── debate/       # Adversarial debate layer
├── extraction/   # Evidence extraction from repos
├── graph/        # LangGraph orchestration
├── ingestion/    # Repo cloning and validation
├── schemas/      # Pydantic v2 typed contracts
├── scoring/      # Rubric evaluation
└── synthesis/    # Report generation
```
