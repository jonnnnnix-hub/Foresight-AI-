"""Post-backtest learning hook — updates adaptive scorer weights.

Called after every backtest run to close the self-learning feedback loop.
Without this, the adaptive scorer uses static weights and never improves.

Flow:
1. Backtest completes → produces trades
2. learning_hook.post_backtest() is called
3. Trades are analyzed, weights updated
4. Updated weights saved to disk
5. Next backtest loads improved weights

This is the CRITICAL final step that was missing — weights were loaded
but never updated from trade outcomes.
"""

from __future__ import annotations

from typing import Any

import structlog

from flowedge.scanner.backtest.adaptive_scorer import (
    REGIME_HISTORICAL_WR,
    STRATEGY_HISTORICAL_WR,
    TICKER_HISTORICAL_WR,
    load_scorer_weights,
    save_scorer_weights,
    update_weights_from_trades,
)

logger = structlog.get_logger()


def post_backtest_learn(
    trades: list[dict[str, Any]],
    model_name: str = "unknown",
    min_trades: int = 10,
) -> dict[str, Any]:
    """Run the self-learning feedback loop after a backtest.

    Updates:
    1. Adaptive scorer weights (feature importance)
    2. Per-ticker historical win rates
    3. Per-strategy historical win rates
    4. Per-regime historical win rates

    Args:
        trades: List of trade dicts from backtest results.
        model_name: Name of the model that produced these trades.
        min_trades: Minimum trades needed to update weights.

    Returns:
        Dict with learning summary (changes made).
    """
    if len(trades) < min_trades:
        logger.info(
            "learning_skipped",
            model=model_name,
            trades=len(trades),
            reason=f"need {min_trades}+ trades",
        )
        return {"skipped": True, "reason": "insufficient_trades"}

    # Load current weights
    weights = load_scorer_weights()
    old_version = weights.version

    # Update weights from trade outcomes
    updated = update_weights_from_trades(trades, weights)

    # Save updated weights
    save_scorer_weights(updated)

    # Compute what changed
    wins = sum(1 for t in trades if float(t.get("pnl_pct", 0)) > 10)
    wr = wins / len(trades) if trades else 0.0

    # Track per-ticker WR updates
    ticker_updates: dict[str, float] = {}
    for ticker, new_wr in TICKER_HISTORICAL_WR.items():
        ticker_updates[ticker] = new_wr

    summary = {
        "model": model_name,
        "trades_analyzed": len(trades),
        "win_rate": round(wr, 3),
        "weights_version": f"v{old_version} → v{updated.version}",
        "trained_on": updated.trained_on_trades,
        "ticker_wr_updates": ticker_updates,
        "strategy_wr": dict(STRATEGY_HISTORICAL_WR),
        "regime_wr": dict(REGIME_HISTORICAL_WR),
    }

    logger.info(
        "learning_complete",
        model=model_name,
        trades=len(trades),
        wr=round(wr, 3),
        version=f"v{updated.version}",
    )

    return summary


def post_backtest_learn_from_result(
    result: Any,
    model_name: str = "unknown",
) -> dict[str, Any]:
    """Convenience wrapper that takes a BacktestResult directly."""
    if not hasattr(result, "trades") or not result.trades:
        return {"skipped": True, "reason": "no_trades"}

    trade_dicts = [t.model_dump(mode="json") for t in result.trades]
    return post_backtest_learn(trade_dicts, model_name)
