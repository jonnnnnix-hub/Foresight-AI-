# FlowEdge Repo Intelligence and Debate Engine

## Mission
Build a production-grade repo analysis platform that ingests public GitHub repositories, extracts structured evidence, runs specialist agent reviews, conducts adversarial debate, scores repositories against a fixed rubric, and produces a final synthesis report.

This is not a toy chatbot.
This is not a vague multi-agent demo.
Build a real, typed, testable system.

## Product goal
The system must:
1. Ingest one or more GitHub repos by URL
2. Clone or inspect them safely
3. Extract structured evidence from:
   - README
   - docs
   - source tree
   - examples
   - tests
   - dependency manifests
   - release and commit metadata when available
4. Run specialist analysis agents on the evidence
5. Run a debate committee where agents challenge each other
6. Score repos against a fixed rubric
7. Produce a final synthesis report for FlowEdge
8. Expose results through CLI, API, and a lightweight UI
9. Persist all runs, evidence, scores, debate rounds, and final reports

## Initial target use case
Compare repos like:
- vectorbt
- NautilusTrader
- Freqtrade
- Qlib
- FinRL
- hftbacktest

The product recommendation must be optimized for:
- short-horizon scalp trading workflows
- trades under 2 hours
- scanner/ranker-style signal discovery
- future support for AI-assisted review and automated execution

## Non-negotiables
- Use Python 3.12
- Use LangGraph for orchestration
- Use Pydantic v2 for schemas
- Use FastAPI for API
- Use Typer for CLI
- Use PostgreSQL for persistence
- Use SQLAlchemy 2.x and Alembic
- Use pytest, ruff, and mypy
- Use structured logging
- Use typed JSON contracts for all agent outputs
- Never use freeform text as the source of truth
- Never execute untrusted repo code by default
- Always preserve evidence, scores, debate records, and reports

## Architecture shape
Build these layers:
1. Ingestion layer
2. Evidence extraction layer
3. Specialist analysis layer
4. Debate layer
5. Judge and synthesis layer
6. Delivery layer

## Core specialist agents
- Repo Cartographer
- Research Engine Analyst
- Execution Realism Analyst
- ML Pipeline Analyst
- Product Architecture Analyst
- Risk and Reliability Analyst
- Skeptic
- Judge

## Build order
1. Project skeleton
2. Core schemas
3. DB models
4. Config and logging
5. CLI and API shell
6. Ingestion pipeline
7. Evidence extraction
8. One specialist agent end-to-end
9. Remaining specialist agents
10. Debate and scoring
11. Reporting
12. UI
13. Tests and hardening

## Engineering rules
- Strong typing everywhere
- Small, composable modules
- No silent failures
- No fake placeholder architecture
- Every agent output must parse into a schema
- Every nontrivial claim must cite evidence
- Prefer code and tests over README marketing
- Separate observed fact, inference, and speculation
- Keep orchestration thin and typed
- Keep providers abstracted from graph logic

## LangGraph expectations
Use a supervisor plus worker subgraph design.

Main graph should include:
- validate_request
- fetch_repo_metadata
- build_evidence_pack
- run_specialists
- aggregate_findings
- debate_round
- score_repos
- synthesize_report
- persist_artifacts
- return_result

## Scoring dimensions
Default weighted dimensions:
- research_power
- execution_realism
- ml_readiness
- productization_value
- reliability
- scalp_fit

Scalp fit must explicitly evaluate:
- intraday data handling
- minute or sub-minute support
- ranking or scanner fit
- liquidity and spread awareness
- time-of-day logic
- order-book or microstructure support where relevant

## Reporting requirements
Every final report must include:
- Executive Summary
- Repo Ranking Table
- Per-Repo Strengths
- Per-Repo Weaknesses
- Debate Highlights
- What Is Real vs Hype
- Borrow / Avoid / Replace Matrix
- Recommended FlowEdge Merged Architecture
- Suggested Build Order for MVP
- Risks and Unknowns
- Appendix with evidence references

## FlowEdge-specific output requirements
The final synthesis must explicitly answer:
1. Which repo is best for:
   - rapid research
   - live execution realism
   - ML workflow
   - productization
   - scalp scanner use case
2. Which components should be borrowed from each repo
3. A proposed merged FlowEdge architecture across:
   - data ingestion
   - feature engineering
   - event labeling
   - signal ranking
   - execution simulation
   - live broker integration
   - monitoring
   - analyst UI
4. MVP recommendation for FlowEdge’s short-horizon scanner product
5. A section titled: do not build this wrong

## Anti-patterns to avoid
Do not:
- build an unstructured agent chat room
- let agents access the full repo unless necessary
- allow unsupported claims without evidence
- produce scores without rationale
- blur extraction and judgment
- rely on README marketing alone
- hide agent disagreement
- optimize prettiness before correctness

## Safety rules
- Do not execute arbitrary repo code during analysis by default
- Do not run install scripts automatically
- Treat cloned repos as untrusted input
- Cap file sizes and binary ingestion
- Redact detected secrets from outputs
- Flag unclear licenses
- Make execution mode explicit and opt-in only

## Required commands before finishing work
Run:
- ruff check .
- mypy src
- pytest

## Delivery expectation
Work in small steps.
Keep files clean.
Run tests often.
Update README as architecture becomes real.
Fix root causes, not symptoms.
