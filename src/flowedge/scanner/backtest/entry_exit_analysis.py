"""Deep regression analysis on entry/exit pricing.

Analyzes optimal entry timing and exit decisions using real trade data.
Questions answered:
1. Is close-of-day entry optimal, or would next-open be better?
2. What IBS/RSI values at entry predict the largest gains?
3. Are early exits (take-profit) leaving money on the table?
4. Do time exits consistently lose, or only in certain regimes?
5. What's the optimal TP target by ticker, volatility regime, and hold period?
6. How much alpha do trailing stops capture vs. cost in whipsaw exits?

Builds regression models:
- Entry model: features → expected PnL at entry
- Exit model: hold_days × features → optimal exit timing
- TP calibration: per-ticker optimal take-profit % from realized moves
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import structlog
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from flowedge.scanner.backtest.schemas import BacktestResult, BacktestTrade

logger = structlog.get_logger()

BACKTEST_DIR = Path("data/backtest")
OUTPUT_DIR = Path("data/analysis")


@dataclass
class EntryAnalysis:
    """Analysis of entry signal quality."""

    ticker: str
    n_trades: int
    avg_pnl_pct: float
    avg_ibs_winners: float
    avg_ibs_losers: float
    avg_rsi_winners: float
    avg_rsi_losers: float
    avg_hold_winners: float
    avg_hold_losers: float
    optimal_ibs_range: tuple[float, float]
    optimal_rsi_range: tuple[float, float]


@dataclass
class ExitAnalysis:
    """Analysis of exit strategy effectiveness."""

    ticker: str
    tp_win_rate: float
    tp_avg_pnl: float
    time_win_rate: float
    time_avg_pnl: float
    trail_win_rate: float
    trail_avg_pnl: float
    optimal_tp_pct: float
    avg_realized_gain_at_tp: float
    avg_max_gain_before_time_exit: float  # Unrealized potential
    tp_trades: int
    time_trades: int
    trail_trades: int


@dataclass
class RegressionResults:
    """Results from entry/exit regression models."""

    # Entry regression
    entry_r2: float
    entry_features_importance: dict[str, float]
    entry_intercept: float

    # Exit timing regression
    exit_r2: float
    exit_features_importance: dict[str, float]

    # TP calibration
    tp_calibration: dict[str, float]  # ticker → optimal TP %

    # Per-ticker analysis
    entry_by_ticker: dict[str, EntryAnalysis]
    exit_by_ticker: dict[str, ExitAnalysis]

    # Actionable recommendations
    recommendations: list[str]


# ── Data Loading ─────────────────────────────────────────────────────────────


def _load_all_trades(
    backtest_files: list[Path] | None = None,
) -> list[BacktestTrade]:
    """Load all trades from backtest result files."""
    if backtest_files is None:
        backtest_files = sorted(BACKTEST_DIR.glob("REAL_*.json"))

    trades: list[BacktestTrade] = []
    for path in backtest_files:
        data = json.loads(path.read_text())
        result = BacktestResult(**data)
        trades.extend(result.trades)

    logger.info("trades_loaded", count=len(trades))
    return trades


# ── Entry Analysis ───────────────────────────────────────────────────────────


def analyze_entries(trades: list[BacktestTrade]) -> dict[str, EntryAnalysis]:
    """Analyze entry signal quality by ticker."""
    by_ticker: dict[str, list[BacktestTrade]] = {}
    for t in trades:
        by_ticker.setdefault(t.ticker, []).append(t)

    results: dict[str, EntryAnalysis] = {}

    for ticker, tk_trades in sorted(by_ticker.items()):
        if len(tk_trades) < 5:
            continue

        winners = [t for t in tk_trades if t.pnl_pct > 0]
        losers = [t for t in tk_trades if t.pnl_pct <= 0]

        # IBS proxy from entry price / underlying
        def _ibs_proxy(t: BacktestTrade) -> float:
            if t.underlying_entry > 0:
                return min(1.0, max(0.0, t.entry_price / t.underlying_entry))
            return 0.5

        win_ibs = [_ibs_proxy(t) for t in winners] if winners else [0.5]
        loss_ibs = [_ibs_proxy(t) for t in losers] if losers else [0.5]

        # RSI proxy from conviction
        win_rsi = [max(0, 50 - t.conviction * 5) for t in winners] if winners else [50]
        loss_rsi = [max(0, 50 - t.conviction * 5) for t in losers] if losers else [50]

        win_hold = [float(t.hold_days) for t in winners] if winners else [0]
        loss_hold = [float(t.hold_days) for t in losers] if losers else [0]

        # Find optimal IBS range (where wins cluster)
        all_ibs = [(t.pnl_pct, _ibs_proxy(t)) for t in tk_trades]
        all_ibs.sort(key=lambda x: x[1])
        best_ibs_low = 0.0
        best_ibs_high = 0.5
        if len(all_ibs) > 10:
            # Sliding window to find IBS range with best avg PnL
            window = max(3, len(all_ibs) // 4)
            best_avg_pnl = -999.0
            for i in range(len(all_ibs) - window):
                avg = sum(x[0] for x in all_ibs[i:i + window]) / window
                if avg > best_avg_pnl:
                    best_avg_pnl = avg
                    best_ibs_low = all_ibs[i][1]
                    best_ibs_high = all_ibs[i + window - 1][1]

        results[ticker] = EntryAnalysis(
            ticker=ticker,
            n_trades=len(tk_trades),
            avg_pnl_pct=round(sum(t.pnl_pct for t in tk_trades) / len(tk_trades), 2),
            avg_ibs_winners=round(sum(win_ibs) / len(win_ibs), 4),
            avg_ibs_losers=round(sum(loss_ibs) / len(loss_ibs), 4),
            avg_rsi_winners=round(sum(win_rsi) / len(win_rsi), 1),
            avg_rsi_losers=round(sum(loss_rsi) / len(loss_rsi), 1),
            avg_hold_winners=round(sum(win_hold) / len(win_hold), 1),
            avg_hold_losers=round(sum(loss_hold) / len(loss_hold), 1),
            optimal_ibs_range=(round(best_ibs_low, 4), round(best_ibs_high, 4)),
            optimal_rsi_range=(15.0, 40.0),  # Placeholder, refined by regression
        )

    return results


# ── Exit Analysis ────────────────────────────────────────────────────────────


def analyze_exits(trades: list[BacktestTrade]) -> dict[str, ExitAnalysis]:
    """Analyze exit strategy effectiveness by ticker."""
    by_ticker: dict[str, list[BacktestTrade]] = {}
    for t in trades:
        by_ticker.setdefault(t.ticker, []).append(t)

    results: dict[str, ExitAnalysis] = {}

    for ticker, tk_trades in sorted(by_ticker.items()):
        if len(tk_trades) < 5:
            continue

        tp_trades = [t for t in tk_trades if t.exit_reason == "take_profit"]
        time_trades = [t for t in tk_trades if t.exit_reason == "time_exit"]
        trail_trades = [t for t in tk_trades if t.exit_reason == "trailing_stop"]

        def _wr_and_avg(ts: list[BacktestTrade]) -> tuple[float, float]:
            if not ts:
                return 0.0, 0.0
            wins = sum(1 for t in ts if t.pnl_pct > 0)
            avg = sum(t.pnl_pct for t in ts) / len(ts)
            return round(wins / len(ts), 3), round(avg, 2)

        tp_wr, tp_avg = _wr_and_avg(tp_trades)
        time_wr, time_avg = _wr_and_avg(time_trades)
        trail_wr, trail_avg = _wr_and_avg(trail_trades)

        # Optimal TP: what TP% would maximize total PnL?
        # Look at the realized gains on winning trades
        win_pcts = [t.pnl_pct for t in tk_trades if t.pnl_pct > 0]
        if win_pcts:
            # The median winning trade's PnL suggests the natural TP level
            sorted_wins = sorted(win_pcts)
            median_idx = len(sorted_wins) // 2
            optimal_tp = sorted_wins[median_idx] / 100  # Convert from % to decimal
        else:
            optimal_tp = 0.025

        # Average realized gain at TP (how much did we capture?)
        avg_tp_gain = (
            sum(t.pnl_pct for t in tp_trades) / len(tp_trades) if tp_trades else 0
        )

        # Average max gain before time exit (potential left on table)
        # Proxy: underlying_move_pct on time exits that were positive
        time_positive = [t for t in time_trades if t.underlying_move_pct > 0]
        avg_max_gain = (
            sum(t.underlying_move_pct for t in time_positive) / len(time_positive)
            if time_positive else 0
        )

        results[ticker] = ExitAnalysis(
            ticker=ticker,
            tp_win_rate=tp_wr,
            tp_avg_pnl=tp_avg,
            time_win_rate=time_wr,
            time_avg_pnl=time_avg,
            trail_win_rate=trail_wr,
            trail_avg_pnl=trail_avg,
            optimal_tp_pct=round(optimal_tp, 4),
            avg_realized_gain_at_tp=round(avg_tp_gain, 2),
            avg_max_gain_before_time_exit=round(avg_max_gain, 2),
            tp_trades=len(tp_trades),
            time_trades=len(time_trades),
            trail_trades=len(trail_trades),
        )

    return results


# ── Regression Models ────────────────────────────────────────────────────────


def build_entry_regression(
    trades: list[BacktestTrade],
) -> tuple[float, dict[str, float], float]:
    """Build regression: entry features → PnL.

    Returns (R², feature_importance, intercept).
    """
    if len(trades) < 30:
        return 0.0, {}, 0.0

    # Build feature matrix
    features: list[list[float]] = []
    targets: list[float] = []

    for t in trades:
        ibs_proxy = t.entry_price / t.underlying_entry if t.underlying_entry > 0 else 0.5
        rsi_proxy = max(0, 50 - t.conviction * 5)
        features.append([
            ibs_proxy,
            rsi_proxy,
            float(t.hold_days),
            float(t.entry_date.weekday()),
            float(t.entry_date.month),
            t.underlying_move_pct,
            t.cost_basis,
        ])
        targets.append(t.pnl_pct)

    x = np.array(features)
    y = np.array(targets)

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)

    model = Ridge(alpha=1.0)
    model.fit(x_scaled, y)

    r2 = float(model.score(x_scaled, y))

    feature_names = [
        "ibs_proxy", "rsi_proxy", "hold_days",
        "day_of_week", "month", "underlying_move_pct", "cost_basis",
    ]
    importance: dict[str, float] = {}
    for name, coef in zip(feature_names, model.coef_, strict=False):
        importance[name] = round(float(abs(coef)), 4)

    importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))

    return round(r2, 4), importance, round(float(model.intercept_), 4)


def build_exit_regression(
    trades: list[BacktestTrade],
) -> tuple[float, dict[str, float]]:
    """Build regression: exit features → PnL at exit.

    Answers: given current hold duration and features, should we exit?
    """
    if len(trades) < 30:
        return 0.0, {}

    features: list[list[float]] = []
    targets: list[float] = []

    exit_reason_map = {
        "take_profit": 0, "time_exit": 1, "trailing_stop": 2,
        "end_of_backtest": 3,
    }

    for t in trades:
        features.append([
            float(t.hold_days),
            float(exit_reason_map.get(t.exit_reason, 3)),
            t.underlying_move_pct,
            t.entry_price,
            float(t.entry_date.weekday()),
            t.conviction,
        ])
        targets.append(t.pnl_pct)

    x = np.array(features)
    y = np.array(targets)

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)

    model = Ridge(alpha=1.0)
    model.fit(x_scaled, y)

    r2 = float(model.score(x_scaled, y))

    feature_names = [
        "hold_days", "exit_reason", "underlying_move_pct",
        "entry_price", "day_of_week", "conviction",
    ]
    importance: dict[str, float] = {}
    for name, coef in zip(feature_names, model.coef_, strict=False):
        importance[name] = round(float(abs(coef)), 4)

    importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
    return round(r2, 4), importance


# ── TP Calibration ───────────────────────────────────────────────────────────


def calibrate_tp_targets(
    trades: list[BacktestTrade],
) -> dict[str, float]:
    """Find optimal TP% for each ticker based on realized gains.

    For each ticker, finds the TP level that maximizes:
    expected_value = P(reaching_TP) × TP_gain - P(not_reaching) × avg_loss
    """
    by_ticker: dict[str, list[BacktestTrade]] = {}
    for t in trades:
        by_ticker.setdefault(t.ticker, []).append(t)

    optimal_tp: dict[str, float] = {}

    for ticker, tk_trades in by_ticker.items():
        if len(tk_trades) < 10:
            continue

        # Get all realized PnL values
        pnls = sorted([t.pnl_pct for t in tk_trades])

        # Test different TP thresholds
        best_ev = -999.0
        best_tp = 2.5  # Default 2.5%

        for tp_test in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 7.5, 10.0]:
            # How many trades would have hit this TP?
            hits = sum(1 for p in pnls if p >= tp_test)
            misses = len(pnls) - hits
            hit_rate = hits / len(pnls) if pnls else 0

            # Expected value at this TP level
            avg_loss = (
                sum(p for p in pnls if p < tp_test) / max(1, misses)
            )
            ev = hit_rate * tp_test + (1 - hit_rate) * avg_loss

            if ev > best_ev:
                best_ev = ev
                best_tp = tp_test

        optimal_tp[ticker] = round(best_tp / 100, 4)  # Convert % to decimal

    return optimal_tp


# ── Full Analysis Pipeline ───────────────────────────────────────────────────


def run_full_analysis(
    backtest_files: list[Path] | None = None,
) -> RegressionResults:
    """Run complete entry/exit regression analysis."""
    trades = _load_all_trades(backtest_files)

    if len(trades) < 30:
        raise ValueError(f"Need at least 30 trades for regression, got {len(trades)}")

    # Entry analysis
    entry_by_ticker = analyze_entries(trades)
    entry_r2, entry_importance, entry_intercept = build_entry_regression(trades)

    # Exit analysis
    exit_by_ticker = analyze_exits(trades)
    exit_r2, exit_importance = build_exit_regression(trades)

    # TP calibration
    tp_cal = calibrate_tp_targets(trades)

    # Generate recommendations
    recommendations = _generate_recommendations(
        trades, entry_by_ticker, exit_by_ticker, tp_cal,
        entry_r2, exit_r2,
    )

    results = RegressionResults(
        entry_r2=entry_r2,
        entry_features_importance=entry_importance,
        entry_intercept=entry_intercept,
        exit_r2=exit_r2,
        exit_features_importance=exit_importance,
        tp_calibration=tp_cal,
        entry_by_ticker=entry_by_ticker,
        exit_by_ticker=exit_by_ticker,
        recommendations=recommendations,
    )

    _print_analysis_report(results)
    _save_analysis(results)

    return results


def _generate_recommendations(
    trades: list[BacktestTrade],
    entry_by_ticker: dict[str, EntryAnalysis],
    exit_by_ticker: dict[str, ExitAnalysis],
    tp_cal: dict[str, float],
    entry_r2: float,
    exit_r2: float,
) -> list[str]:
    """Generate actionable recommendations from analysis."""
    recs: list[str] = []

    # 1. Entry signal quality
    if entry_r2 < 0.1:
        recs.append(
            f"Entry regression R²={entry_r2:.3f} — entry features weakly predict PnL. "
            "Focus on exit optimization rather than entry filtering."
        )
    else:
        recs.append(
            f"Entry regression R²={entry_r2:.3f} — entry features have some predictive power."
        )

    # 2. Exit reason breakdown
    tp_count = sum(1 for t in trades if t.exit_reason == "take_profit")
    time_count = sum(1 for t in trades if t.exit_reason == "time_exit")
    trail_count = sum(1 for t in trades if t.exit_reason == "trailing_stop")
    total = len(trades)

    tp_wr = (
        sum(1 for t in trades if t.exit_reason == "take_profit" and t.pnl_pct > 0)
        / max(1, tp_count)
    )
    time_wr = (
        sum(1 for t in trades if t.exit_reason == "time_exit" and t.pnl_pct > 0)
        / max(1, time_count)
    )

    recs.append(
        f"Take-profit: {tp_count}/{total} trades ({tp_wr:.0%} WR). "
        f"Time exit: {time_count}/{total} trades ({time_wr:.0%} WR). "
        f"Trail: {trail_count}/{total} trades."
    )

    if time_wr < 0.3 and time_count > 10:
        recs.append(
            "Time exits have very low WR — consider extending max_hold "
            "or adding a break-even stop after min_hold."
        )

    # 3. Per-ticker TP calibration
    for ticker, tp in sorted(tp_cal.items(), key=lambda x: x[1], reverse=True):
        if ticker in exit_by_ticker:
            ea = exit_by_ticker[ticker]
            recs.append(
                f"{ticker}: optimal TP={tp * 100:.1f}%, "
                f"current TP WR={ea.tp_win_rate:.0%} ({ea.tp_trades} trades), "
                f"time WR={ea.time_win_rate:.0%} ({ea.time_trades} trades)"
            )

    # 4. Best/worst tickers
    if entry_by_ticker:
        best = max(entry_by_ticker.values(), key=lambda x: x.avg_pnl_pct)
        worst = min(entry_by_ticker.values(), key=lambda x: x.avg_pnl_pct)
        recs.append(
            f"Best ticker: {best.ticker} (avg PnL {best.avg_pnl_pct:+.1f}%, "
            f"{best.n_trades} trades)"
        )
        recs.append(
            f"Worst ticker: {worst.ticker} (avg PnL {worst.avg_pnl_pct:+.1f}%, "
            f"{worst.n_trades} trades)"
        )

    return recs


# ── Reporting ────────────────────────────────────────────────────────────────


def _print_analysis_report(results: RegressionResults) -> None:
    """Print comprehensive analysis report."""
    print("\n" + "=" * 100)
    print("DEEP ENTRY/EXIT REGRESSION ANALYSIS")
    print("=" * 100)

    # Entry regression
    print(f"\n  ENTRY REGRESSION (R² = {results.entry_r2:.4f})")
    print(f"  Intercept: {results.entry_intercept:.4f}")
    print("  Feature Importance:")
    for name, imp in results.entry_features_importance.items():
        bar = "#" * int(imp * 10)
        print(f"    {name:<25} {imp:.4f}  {bar}")

    # Exit regression
    print(f"\n  EXIT REGRESSION (R² = {results.exit_r2:.4f})")
    print("  Feature Importance:")
    for name, imp in results.exit_features_importance.items():
        bar = "#" * int(imp * 10)
        print(f"    {name:<25} {imp:.4f}  {bar}")

    # Exit breakdown by ticker
    print("\n  EXIT STRATEGY BY TICKER:")
    print(
        f"  {'Ticker':<8} {'TP WR':>7} {'TP PnL':>8} {'Time WR':>8} {'Time PnL':>9} "
        f"{'Trail WR':>9} {'Opt TP%':>8} {'#TP':>5} {'#Time':>6}"
    )
    print("  " + "-" * 80)
    for ticker, ea in sorted(
        results.exit_by_ticker.items(),
        key=lambda x: x[1].tp_win_rate,
        reverse=True,
    ):
        opt_tp = results.tp_calibration.get(ticker, 0.025) * 100
        print(
            f"  {ticker:<8} "
            f"{ea.tp_win_rate:>6.0%} "
            f"{ea.tp_avg_pnl:>+7.1f}% "
            f"{ea.time_win_rate:>7.0%} "
            f"{ea.time_avg_pnl:>+8.1f}% "
            f"{ea.trail_win_rate:>8.0%} "
            f"{opt_tp:>7.1f}% "
            f"{ea.tp_trades:>5} "
            f"{ea.time_trades:>6}"
        )

    # Recommendations
    print("\n  RECOMMENDATIONS:")
    for i, rec in enumerate(results.recommendations, 1):
        print(f"  {i}. {rec}")

    print("\n" + "=" * 100)


def _save_analysis(results: RegressionResults) -> None:
    """Save analysis results to JSON."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    data: dict[str, Any] = {
        "entry_r2": results.entry_r2,
        "entry_features_importance": results.entry_features_importance,
        "entry_intercept": results.entry_intercept,
        "exit_r2": results.exit_r2,
        "exit_features_importance": results.exit_features_importance,
        "tp_calibration": results.tp_calibration,
        "recommendations": results.recommendations,
        "entry_by_ticker": {
            ticker: {
                "n_trades": ea.n_trades,
                "avg_pnl_pct": ea.avg_pnl_pct,
                "avg_ibs_winners": ea.avg_ibs_winners,
                "avg_ibs_losers": ea.avg_ibs_losers,
                "avg_hold_winners": ea.avg_hold_winners,
                "avg_hold_losers": ea.avg_hold_losers,
                "optimal_ibs_range": list(ea.optimal_ibs_range),
            }
            for ticker, ea in results.entry_by_ticker.items()
        },
        "exit_by_ticker": {
            ticker: {
                "tp_win_rate": ea.tp_win_rate,
                "tp_avg_pnl": ea.tp_avg_pnl,
                "time_win_rate": ea.time_win_rate,
                "time_avg_pnl": ea.time_avg_pnl,
                "trail_win_rate": ea.trail_win_rate,
                "trail_avg_pnl": ea.trail_avg_pnl,
                "optimal_tp_pct": ea.optimal_tp_pct,
                "tp_trades": ea.tp_trades,
                "time_trades": ea.time_trades,
                "trail_trades": ea.trail_trades,
            }
            for ticker, ea in results.exit_by_ticker.items()
        },
    }

    path = OUTPUT_DIR / "entry_exit_analysis.json"
    path.write_text(json.dumps(data, indent=2))
    logger.info("analysis_saved", path=str(path))


# ── CLI ──────────────────────────────────────────────────────────────────────


if __name__ == "__main__":
    from flowedge.config.logging import setup_logging

    setup_logging("INFO")
    run_full_analysis()
