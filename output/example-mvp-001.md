# FlowEdge Analysis Report — example-mvp-001

*Generated: 2026-04-11 01:25:46.424852+00:00*

## Executive Summary

Analysis of 3 repositories complete. Top-ranked: nautilus_trader (score: 6.9). Best dimensions: execution_realism, productization_value.

## Repo Ranking

| Rank | Repository | Score | Best For | Weakest At |
|------|-----------|-------|----------|------------|
| 1 | nautilus_trader | 6.9 | execution_realism, productization_value | ml_readiness, research_power |
| 2 | freqtrade | 6.67 | reliability, productization_value | ml_readiness, scalp_fit |
| 3 | vectorbt | 5.1 | research_power, scalp_fit | execution_realism, ml_readiness |

## Borrow / Avoid / Replace Matrix

### vectorbt

**Borrow:**
- Vectorized backtesting engine for research iteration
- Parameter sweep infrastructure
- Portfolio analytics visualization

**Avoid:**
- Execution model (no realism)
- Live trading path (nonexistent)

**Replace:**
- Data pipeline (too tightly coupled to pandas)

### nautilus_trader

**Borrow:**
- Event-driven execution engine
- Order management and fill simulation
- Broker adapter architecture
- Risk management framework

**Avoid:**
- Complexity overhead for simple strategies
- Steep onboarding curve

**Replace:**
- ML integration (minimal)
- Scanner/ranker infrastructure (missing)

### freqtrade

**Borrow:**
- Bot management and deployment infrastructure
- Strategy plugin system
- Hyperopt integration
- Community-tested exchange adapters

**Avoid:**
- Timeframe assumptions (not sub-minute)
- Monolithic strategy class

**Replace:**
- Execution simulation (too simplified for scalps)
- Feature engineering pipeline (basic)

## Debate Highlights

- [execution_realism (vectorbt vs nautilus_trader)] execution_analyst: vectorbt prioritizes research speed over execution realism. For scalp trading, this is a critical gap.
- [scalp_trading_fitness (freqtrade vs nautilus_trader)] skeptic: freqtrade's architecture favors swing trading. nautilus_trader has better sub-minute execution support.

## What Is Real vs Hype

- **vectorbt**: More real than hype
- **nautilus_trader**: More real than hype
- **freqtrade**: More real than hype

## Recommended FlowEdge Merged Architecture

| Component | Source |
|-----------|--------|
| data_ingestion | nautilus_trader (event-driven data handlers) |
| feature_engineering | custom (borrow vectorbt analytics) |
| event_labeling | custom (short-horizon labels) |
| signal_ranking | custom (ML-based scanner) |
| execution_simulation | nautilus_trader (realistic fills) |
| live_broker_integration | nautilus_trader + freqtrade adapters |
| monitoring | custom (borrow freqtrade bot management) |
| analyst_ui | custom (borrow vectorbt visualization) |

## Suggested Build Order for MVP

1. Data ingestion layer using nautilus_trader patterns
2. Feature engineering with vectorbt-style analytics
3. Signal scanner/ranker with ML scoring
4. Execution simulation with nautilus_trader fills
5. Paper trading integration
6. Monitoring dashboard
7. Live broker connection

## Do Not Build This Wrong

Do not build a backtester that assumes zero slippage
Do not build a scanner without time-of-day awareness
Do not ship without walk-forward validation
Do not treat freqtrade timeframes as sub-minute capable
Do not build a monolithic strategy class

## Risks and Unknowns

- nautilus_trader Rust core may limit Python extensibility
- vectorbt is single-maintainer — bus factor risk
- freqtrade exchange adapters may not support sub-minute data
- ML pipeline for real-time scoring needs custom work
- Latency requirements for scalp trading are untested

## Appendix: Evidence References

- vectorbt: src/vectorbt/portfolio/base.py
- nautilus_trader: nautilus_trader/execution/engine.pyx
- freqtrade: freqtrade/strategy/interface.py
