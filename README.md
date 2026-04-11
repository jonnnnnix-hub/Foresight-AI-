# FlowEdge Repo Intelligence Engine

Production-grade repo analysis platform that ingests GitHub repositories, extracts structured evidence, runs specialist agent reviews, conducts adversarial debate, scores repos against a fixed rubric, and produces a final synthesis report.

## Target Use Case

Compare trading system repos (vectorbt, NautilusTrader, Freqtrade, Qlib, FinRL, hftbacktest) optimized for short-horizon scalp trading workflows.

## Architecture

```
Ingestion → Evidence Extraction → Specialist Agents → Debate → Scoring → Synthesis Report
```

**Layers:**
1. **Ingestion** — Clone and validate GitHub repos safely (depth=1, untrusted-code-safe)
2. **Extraction** — Scan files, classify evidence, build typed evidence packs
3. **Specialists** — 7 analyst agents + 1 judge evaluate different dimensions
4. **Debate** — Adversarial rounds where agents challenge each other's findings
5. **Scoring** — Weighted rubric across 6 dimensions with evidence-backed rationale
6. **Synthesis** — Final report with rankings, borrow/avoid/replace matrix, and merged architecture

## Stack

- Python 3.12, LangGraph, Pydantic v2
- FastAPI + Typer + SQLAlchemy 2.x + Alembic
- PostgreSQL, structlog, pytest, ruff, mypy
- Anthropic Claude for specialist agent analysis

## Setup

```bash
# Install uv (if needed)
pip install uv

# Install Python 3.12
uv python install 3.12

# Create virtual environment
uv venv .venv --python 3.12
source .venv/bin/activate

# Install dependencies
uv pip install -e ".[dev]" --python .venv/bin/python

# Copy environment config
cp .env.example .env
# Edit .env and set ANTHROPIC_API_KEY
```

## Running the Comparison

### CLI

```bash
# Analyze repos
flowedge analyze \
    https://github.com/polakowo/vectorbt \
    https://github.com/nautechsystems/nautilus_trader \
    https://github.com/freqtrade/freqtrade \
    --output-dir ./output

# Or use the example script
./examples/compare_trading_repos.sh
```

### API

```bash
# Start server
uvicorn flowedge.api.app:app --reload

# Submit analysis
curl -X POST http://localhost:8000/api/v1/analyze \
  -H "Content-Type: application/json" \
  -d '{"repo_urls": ["https://github.com/polakowo/vectorbt", "https://github.com/nautechsystems/nautilus_trader"]}'

# Check status / get results
curl http://localhost:8000/api/v1/runs/{run_id}
```

### Example Reports (No API Key Needed)

```bash
PYTHONPATH=src python examples/generate_example_report.py
# Produces output/example-mvp-001.json and output/example-mvp-001.md
```

## Scoring Dimensions

| Dimension | Weight | Evaluated By |
|-----------|--------|-------------|
| Research Power | 0.15 | Research Analyst |
| Execution Realism | 0.20 | Execution Analyst |
| ML Readiness | 0.15 | ML Analyst |
| Productization Value | 0.15 | Product Architect |
| Reliability | 0.15 | Risk Analyst |
| Scalp Fit | 0.20 | All (cross-cutting) |

## Specialist Agents

| Agent | Purpose |
|-------|---------|
| Repo Cartographer | Maps structure, modules, entry points, architecture seams |
| Research Analyst | Evaluates backtesting, parameter sweeps, scanner suitability |
| Execution Analyst | Evaluates slippage modeling, fills, broker integration |
| ML Analyst | Evaluates feature pipelines, leakage controls, inference design |
| Product Architect | Evaluates modularity, config, API fitness, extensibility |
| Risk Analyst | Evaluates test maturity, dependency health, failure modes |
| Skeptic | Challenges overclaims, finds hype without implementation |
| Judge | Synthesizes all findings into final ranking and recommendations |

## Quality Checks

```bash
ruff check src/ tests/
mypy src/
pytest tests/ -q
```

## Project Structure

```
src/flowedge/
├── agents/       # 7 specialist analysts + judge + LLM abstraction
├── api/          # FastAPI routes (health, analyze, run status)
├── cli/          # Typer CLI (analyze, version)
├── config/       # Settings and structured logging
├── db/           # SQLAlchemy 2.x models and async sessions
├── debate/       # Adversarial debate engine with contradiction detection
├── extraction/   # File scanner and evidence classification
├── graph/        # LangGraph pipeline (state, nodes, orchestration)
├── ingestion/    # Safe repo cloning with URL validation
├── schemas/      # Pydantic v2 contracts (evidence, scoring, debate, report, agents)
├── scoring/      # Weighted rubric computation
└── synthesis/    # Report generation and export (JSON + Markdown)
```
