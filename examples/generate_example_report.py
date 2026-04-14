"""Generate example JSON and Markdown reports from mock data.

This script creates realistic example reports without calling the LLM,
useful for testing the export pipeline and understanding report structure.

Usage:
    python examples/generate_example_report.py
"""

from pathlib import Path

from flowedge.schemas.agents import JudgeOutput
from flowedge.schemas.debate import (
    DebateEntry,
    DebateRecord,
    DebateRound,
    DebateStance,
)
from flowedge.schemas.scoring import DimensionScore, RepoScorecard
from flowedge.synthesis.engine import build_synthesis_report
from flowedge.synthesis.export import export_json, export_markdown


def make_scorecard(name: str, scores: dict[str, float]) -> RepoScorecard:
    """Create a scorecard with the given dimension scores."""
    weights = {
        "research_power": 0.15,
        "execution_realism": 0.20,
        "ml_readiness": 0.15,
        "productization_value": 0.15,
        "reliability": 0.15,
        "scalp_fit": 0.20,
    }
    dims = [
        DimensionScore(
            dimension=dim,
            score=scores.get(dim, 5.0),
            weight=weights[dim],
            rationale=f"Score for {dim} based on evidence analysis",
        )
        for dim in weights
    ]
    return RepoScorecard(
        repo_name=name,
        repo_url=f"https://github.com/example/{name}",
        dimensions=dims,
    )


def main() -> None:
    # Realistic scores based on the target repos
    vectorbt_card = make_scorecard("vectorbt", {
        "research_power": 8.5,
        "execution_realism": 3.0,
        "ml_readiness": 4.0,
        "productization_value": 5.5,
        "reliability": 4.0,
        "scalp_fit": 6.0,
    })
    nautilus_card = make_scorecard("nautilus_trader", {
        "research_power": 6.0,
        "execution_realism": 9.0,
        "ml_readiness": 3.5,
        "productization_value": 7.5,
        "reliability": 7.0,
        "scalp_fit": 7.5,
    })
    freqtrade_card = make_scorecard("freqtrade", {
        "research_power": 7.0,
        "execution_realism": 6.5,
        "ml_readiness": 5.0,
        "productization_value": 8.0,
        "reliability": 8.5,
        "scalp_fit": 5.5,
    })

    scorecards = sorted(
        [vectorbt_card, nautilus_card, freqtrade_card],
        key=lambda c: c.weighted_total,
        reverse=True,
    )

    debate = DebateRecord(
        repo_names=["vectorbt", "nautilus_trader", "freqtrade"],
        rounds=[
            DebateRound(
                round_number=1,
                topic="execution_realism (vectorbt vs nautilus_trader)",
                entries=[
                    DebateEntry(
                        agent_name="research_analyst",
                        stance=DebateStance.SUPPORT,
                        claim="vectorbt has superior research iteration speed",
                        argument="Vectorized backtesting enables rapid parameter sweeps",
                        confidence=0.85,
                    ),
                    DebateEntry(
                        agent_name="execution_analyst",
                        stance=DebateStance.CHALLENGE,
                        claim="vectorbt execution model is unrealistic",
                        argument="No slippage, no order book, no partial fills",
                        confidence=0.90,
                    ),
                ],
                resolution=(
                    "execution_analyst: vectorbt prioritizes research speed over "
                    "execution realism. For scalp trading, this is a critical gap."
                ),
            ),
            DebateRound(
                round_number=2,
                topic="scalp_trading_fitness (freqtrade vs nautilus_trader)",
                entries=[
                    DebateEntry(
                        agent_name="product_architect",
                        stance=DebateStance.SUPPORT,
                        claim="freqtrade is more production-ready",
                        argument="Mature plugin system, bot management, live trading",
                        confidence=0.80,
                    ),
                    DebateEntry(
                        agent_name="skeptic",
                        stance=DebateStance.CHALLENGE,
                        claim="freqtrade scalp support is weak",
                        argument="Designed for longer holds, not sub-2h scalps",
                        confidence=0.75,
                    ),
                ],
                resolution=(
                    "skeptic: freqtrade's architecture favors swing trading. "
                    "nautilus_trader has better sub-minute execution support."
                ),
            ),
        ],
    )

    judge = JudgeOutput(
        repo_ranking=["nautilus_trader", "freqtrade", "vectorbt"],
        top_repo_by_category={
            "rapid_research": "vectorbt",
            "live_execution_realism": "nautilus_trader",
            "ml_workflow": "freqtrade",
            "productization": "freqtrade",
            "scalp_scanner": "nautilus_trader",
        },
        borrow_avoid_replace={
            "vectorbt": {
                "borrow": [
                    "Vectorized backtesting engine for research iteration",
                    "Parameter sweep infrastructure",
                    "Portfolio analytics visualization",
                ],
                "avoid": [
                    "Execution model (no realism)",
                    "Live trading path (nonexistent)",
                ],
                "replace": [
                    "Data pipeline (too tightly coupled to pandas)",
                ],
            },
            "nautilus_trader": {
                "borrow": [
                    "Event-driven execution engine",
                    "Order management and fill simulation",
                    "Broker adapter architecture",
                    "Risk management framework",
                ],
                "avoid": [
                    "Complexity overhead for simple strategies",
                    "Steep onboarding curve",
                ],
                "replace": [
                    "ML integration (minimal)",
                    "Scanner/ranker infrastructure (missing)",
                ],
            },
            "freqtrade": {
                "borrow": [
                    "Bot management and deployment infrastructure",
                    "Strategy plugin system",
                    "Hyperopt integration",
                    "Community-tested exchange adapters",
                ],
                "avoid": [
                    "Timeframe assumptions (not sub-minute)",
                    "Monolithic strategy class",
                ],
                "replace": [
                    "Execution simulation (too simplified for scalps)",
                    "Feature engineering pipeline (basic)",
                ],
            },
        },
        merged_architecture={
            "data_ingestion": "nautilus_trader (event-driven data handlers)",
            "feature_engineering": "custom (borrow vectorbt analytics)",
            "event_labeling": "custom (short-horizon labels)",
            "signal_ranking": "custom (ML-based scanner)",
            "execution_simulation": "nautilus_trader (realistic fills)",
            "live_broker_integration": "nautilus_trader + freqtrade adapters",
            "monitoring": "custom (borrow freqtrade bot management)",
            "analyst_ui": "custom (borrow vectorbt visualization)",
        },
        mvp_build_order=[
            "1. Data ingestion layer using nautilus_trader patterns",
            "2. Feature engineering with vectorbt-style analytics",
            "3. Signal scanner/ranker with ML scoring",
            "4. Execution simulation with nautilus_trader fills",
            "5. Paper trading integration",
            "6. Monitoring dashboard",
            "7. Live broker connection",
        ],
        risks_and_unknowns=[
            "nautilus_trader Rust core may limit Python extensibility",
            "vectorbt is single-maintainer — bus factor risk",
            "freqtrade exchange adapters may not support sub-minute data",
            "ML pipeline for real-time scoring needs custom work",
            "Latency requirements for scalp trading are untested",
        ],
        do_not_build_wrong=[
            "Do not build a backtester that assumes zero slippage",
            "Do not build a scanner without time-of-day awareness",
            "Do not ship without walk-forward validation",
            "Do not treat freqtrade timeframes as sub-minute capable",
            "Do not build a monolithic strategy class",
        ],
        evidence_refs=[
            "vectorbt: src/vectorbt/portfolio/base.py",
            "nautilus_trader: nautilus_trader/execution/engine.pyx",
            "freqtrade: freqtrade/strategy/interface.py",
        ],
    )

    report = build_synthesis_report(
        run_id="example-mvp-001",
        scorecards=scorecards,
        debate_record=debate,
        judge_output=judge,
    )

    out = Path("output")
    json_path = export_json(report, out / "example-mvp-001.json")
    md_path = export_markdown(report, out / "example-mvp-001.md")

    print(f"JSON report: {json_path}")
    print(f"Markdown report: {md_path}")
    print(f"\nTop-ranked: {report.rankings[0].repo_name} "
          f"(score: {report.rankings[0].weighted_score})")


if __name__ == "__main__":
    main()
