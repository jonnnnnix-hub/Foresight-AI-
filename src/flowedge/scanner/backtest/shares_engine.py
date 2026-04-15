"""FlowEdge Shares Engine — unified share trading backtester.

Supports Precision, Hybrid, and Scalp modes on actual stock shares.
No options, no theta, no expiration. $0.18/share rebate (Public.com).

Shares beat options on our IBS reversion signals:
- Precision (shares): 54.8% WR, +385% total, 493 trades
- Options equivalent: 50% WR, -7% (theta kills multi-day holds)

Modes:
- PRECISION: IBS<0.20, RSI<45, 2.5% TP, 12d hold, 3d min hold
- HYBRID: IBS<0.20, RSI<45, 2.5% TP, 7d hold, 3d min hold
- SCALP_SHARES: IBS<0.10, RSI<20, 0.5% TP, 2d hold, 0d min hold
- SCALP_OPTIONS: Same signals, but executes via real option contracts
"""

from __future__ import annotations

import json
import uuid
from datetime import date
from enum import StrEnum
from math import sqrt
from pathlib import Path
from typing import Any

import structlog

from flowedge.scanner.backtest.schemas import (
    BacktestResult,
    BacktestTrade,
    TradeOutcome,
)

logger = structlog.get_logger()

CACHE_DIR = Path("data/flat_files_s3")
REBATE_PER_SHARE = 0.18  # Public.com rebate


class TradeMode(StrEnum):
    PRECISION_SHARES = "precision_shares"
    HYBRID_SHARES = "hybrid_shares"
    SCALP_SHARES = "scalp_shares"
    SCALP_OPTIONS = "scalp_options"


# Model configurations
MODELS: dict[str, dict[str, Any]] = {
    "precision_shares": {
        "tickers": [
            "META", "AVGO", "SPY", "GOOGL", "NVDA", "AMZN",
            "INTC", "DIA", "AAPL", "QQQ", "IWM", "PLTR",
        ],
        "ibs_threshold": 0.20,
        "rsi_threshold": 45.0,
        "tp_pct": 0.025,       # 2.5% take profit
        "max_hold": 12,
        "min_hold": 3,
        "trail_trigger": 0.015,  # Trail activates at 1.5% gain
        "trail_exit": 0.005,    # Exit if drops below 0.5% gain
        "risk_pct": 0.10,       # 10% of portfolio per trade
        "cooldown": 3,
        "max_positions": 4,
    },
    "hybrid_shares": {
        "tickers": [
            "META", "AVGO", "NVDA", "AMZN", "SPY", "DIA",
            "AAPL", "QQQ", "IWM", "XLK",
        ],
        "ibs_threshold": 0.20,
        "rsi_threshold": 45.0,
        "tp_pct": 0.025,
        "max_hold": 7,
        "min_hold": 3,
        "trail_trigger": 0.015,
        "trail_exit": 0.005,
        "risk_pct": 0.10,
        "cooldown": 3,
        "max_positions": 4,
    },
    "scalp_shares": {
        "tickers": [
            "AVGO", "META", "XLK", "NVDA", "AMZN", "NFLX",
            "SPY", "AMD", "INTC", "PLTR",
        ],
        "ibs_threshold": 0.10,
        "rsi_threshold": 20.0,
        "tp_pct": 0.005,        # 0.5% take profit (scalp)
        "max_hold": 2,
        "min_hold": 0,
        "trail_trigger": 0.003,
        "trail_exit": 0.001,
        "risk_pct": 0.05,
        "cooldown": 1,
        "max_positions": 3,
    },
}


def _gf(bar: dict[str, Any], long_key: str, short_key: str) -> float:
    val: object = bar.get(long_key, bar.get(short_key, 0))
    return float(val)  # type: ignore[arg-type]


def _load_daily(ticker: str) -> list[dict[str, Any]]:
    min_dir = CACHE_DIR / ticker / "1min"
    if not min_dir.exists():
        return []
    bars: list[dict[str, Any]] = []
    for f in sorted(min_dir.glob("*.json")):
        bars.extend(json.loads(f.read_text()))
    daily: dict[str, dict[str, Any]] = {}
    for b in bars:
        d = str(b.get("date", b.get("d", "")))
        if not d:
            continue
        o = _gf(b, "open", "o")
        h = _gf(b, "high", "h")
        lo = _gf(b, "low", "l")
        c = _gf(b, "close", "c")
        v = _gf(b, "volume", "v")
        if d not in daily:
            daily[d] = {"date": d, "open": o, "high": h, "low": lo, "close": c, "volume": v}
        else:
            daily[d]["high"] = max(daily[d]["high"], h)
            if lo > 0:
                daily[d]["low"] = min(daily[d]["low"], lo)
            daily[d]["close"] = c
            daily[d]["volume"] += v
    return [daily[d] for d in sorted(daily)]


def _rsi(closes: list[float], period: int = 14) -> float:
    if len(closes) < period + 1:
        return 50.0
    gain_sum, loss_sum = 0.0, 0.0
    for i in range(-period, 0):
        d = closes[i] - closes[i - 1]
        if d > 0:
            gain_sum += d
        else:
            loss_sum += abs(d)
    ag = gain_sum / period
    al = loss_sum / period
    if al == 0:
        return 100.0
    return 100.0 - 100.0 / (1.0 + ag / al)


def run_shares_backtest(
    mode: str = "precision_shares",
    starting_capital: float = 10_000.0,
    tickers: list[str] | None = None,
    params: dict[str, Any] | None = None,
    _record: bool = True,
) -> BacktestResult:
    """Run share trading backtest on cached minute data.

    No options. No theta. No spreads. Real stock prices.
    $0.18/share rebate on every trade.

    Args:
        params: Optional parameter overrides. Keys match MODELS dict entries
                (ibs_threshold, rsi_threshold, tp_pct, max_hold, min_hold,
                trail_trigger, trail_exit, risk_pct, cooldown, max_positions).
    """
    config = MODELS.get(mode, MODELS["precision_shares"])
    tickers = tickers or config["tickers"]
    overrides = params or {}

    ibs_thresh = overrides.get("ibs_threshold", config["ibs_threshold"])
    rsi_thresh = overrides.get("rsi_threshold", config["rsi_threshold"])
    tp_pct = overrides.get("tp_pct", config["tp_pct"])
    max_hold = overrides.get("max_hold", config["max_hold"])
    min_hold = overrides.get("min_hold", config["min_hold"])
    trail_trigger = overrides.get("trail_trigger", config["trail_trigger"])
    trail_exit = overrides.get("trail_exit", config["trail_exit"])
    risk_pct = overrides.get("risk_pct", config["risk_pct"])
    cooldown = overrides.get("cooldown", config["cooldown"])
    max_positions = overrides.get("max_positions", config["max_positions"])

    # Load data
    all_daily: dict[str, list[dict[str, Any]]] = {}
    for ticker in tickers:
        daily = _load_daily(ticker)
        if len(daily) >= 60:
            all_daily[ticker] = daily

    if not all_daily:
        return BacktestResult(
            run_id=f"{mode}-{uuid.uuid4().hex[:6]}",
            tickers=tickers, starting_capital=starting_capital,
        )

    # Get all dates
    all_dates: set[str] = set()
    for bars in all_daily.values():
        for b in bars:
            all_dates.add(b["date"])
    sorted_dates = sorted(all_dates)

    cash = starting_capital
    positions: list[dict[str, Any]] = []
    closed: list[BacktestTrade] = []
    daily_values: list[tuple[str, float]] = []
    last_entry: dict[str, int] = {}

    for day_idx, d in enumerate(sorted_dates):
        # Update + check exits on open positions
        to_close: list[tuple[dict[str, Any], str]] = []
        for pos in positions:
            tk_bars = all_daily.get(pos["ticker"], [])
            day_bar = next((b for b in tk_bars if b["date"] == d), None)
            if not day_bar:
                continue

            pos["days_held"] += 1
            pos["current_price"] = day_bar["close"]
            pos["max_price"] = max(pos["max_price"], day_bar["high"])

            gain = (day_bar["close"] - pos["entry_price"]) / pos["entry_price"]
            max_gain = (pos["max_price"] - pos["entry_price"]) / pos["entry_price"]

            # TP
            if gain >= tp_pct:
                to_close.append((pos, "take_profit"))
                continue

            if pos["days_held"] < min_hold:
                continue

            # Time exit
            if pos["days_held"] >= max_hold:
                to_close.append((pos, "time_exit"))
                continue

            # Trail
            if max_gain >= trail_trigger and gain < trail_exit:
                to_close.append((pos, "trailing_stop"))
                continue

        for pos, reason in to_close:
            if pos not in positions:
                continue
            shares = pos["shares"]
            exit_price = pos["current_price"]
            pnl = (exit_price - pos["entry_price"]) * shares
            rebate = shares * REBATE_PER_SHARE * 2  # Buy + sell
            net_pnl = pnl + rebate
            pnl_pct = net_pnl / pos["cost_basis"] * 100

            cash += pos["cost_basis"] + net_pnl
            positions.remove(pos)

            outcome = TradeOutcome.WIN if pnl_pct > 0.5 else (
                TradeOutcome.LOSS if pnl_pct < -0.5 else TradeOutcome.BREAKEVEN
            )

            closed.append(BacktestTrade(
                ticker=pos["ticker"],
                entry_date=date.fromisoformat(pos["entry_date"]),
                exit_date=date.fromisoformat(d),
                option_type="shares",
                strike=0,
                entry_price=round(pos["entry_price"], 2),
                exit_price=round(exit_price, 2),
                underlying_entry=round(pos["entry_price"], 2),
                underlying_exit=round(exit_price, 2),
                underlying_move_pct=round(
                    (exit_price - pos["entry_price"]) / pos["entry_price"] * 100, 2,
                ),
                pnl_per_contract=round(net_pnl / shares, 2),
                pnl_pct=round(pnl_pct, 2),
                outcome=outcome,
                signal_score=9.0,
                signal_type=f"{mode}|{shares}sh",
                hold_days=pos["days_held"],
                strategy=mode,
                regime="uptrend",
                conviction=9.0,
                exit_reason=reason,
                contracts=shares,
                cost_basis=round(pos["cost_basis"], 2),
                exit_value=round(pos["cost_basis"] + net_pnl, 2),
            ))

        # Scan for entries
        if len(positions) < max_positions:
            for ticker in all_daily:
                if len(positions) >= max_positions:
                    break
                if any(p["ticker"] == ticker for p in positions):
                    continue
                if day_idx - last_entry.get(ticker, -999) < cooldown:
                    continue

                tk_bars = all_daily[ticker]
                idx = next(
                    (i for i, b in enumerate(tk_bars) if b["date"] == d), None,
                )
                if idx is None or idx < 55:
                    continue

                bar = tk_bars[idx]
                closes = [tk_bars[j]["close"] for j in range(max(0, idx - 49), idx + 1)]
                if len(closes) < 50:
                    continue

                # Uptrend filter
                sma20 = sum(closes[-20:]) / 20
                sma50 = sum(closes[-50:]) / 50
                if sma20 <= sma50:
                    continue

                # IBS filter
                rng = bar["high"] - bar["low"]
                if rng <= 0 or bar["close"] <= 0:
                    continue
                ibs = (bar["close"] - bar["low"]) / rng
                if ibs >= ibs_thresh:
                    continue

                # RSI filter
                r = _rsi(closes)
                if r >= rsi_thresh:
                    continue

                # Prior-day down
                if idx > 0 and bar["close"] >= tk_bars[idx - 1]["close"]:
                    continue

                # BUY SHARES
                entry_price = bar["close"]
                budget = cash * risk_pct
                shares = int(budget / entry_price)
                if shares < 1:
                    continue
                cost = shares * entry_price

                if cost > cash * 0.95:
                    continue

                cash -= cost
                positions.append({
                    "ticker": ticker,
                    "entry_date": d,
                    "entry_price": entry_price,
                    "shares": shares,
                    "cost_basis": cost,
                    "days_held": 0,
                    "current_price": entry_price,
                    "max_price": entry_price,
                })
                last_entry[ticker] = day_idx

        # Daily snapshot
        pos_value = sum(p["current_price"] * p["shares"] for p in positions)
        daily_values.append((d, cash + pos_value))

    # Close remaining
    for pos in positions[:]:
        exit_price = pos["current_price"]
        shares = pos["shares"]
        pnl = (exit_price - pos["entry_price"]) * shares
        rebate = shares * REBATE_PER_SHARE * 2
        net_pnl = pnl + rebate
        pnl_pct = net_pnl / pos["cost_basis"] * 100
        cash += pos["cost_basis"] + net_pnl

        closed.append(BacktestTrade(
            ticker=pos["ticker"],
            entry_date=date.fromisoformat(pos["entry_date"]),
            exit_date=date.fromisoformat(sorted_dates[-1]),
            option_type="shares", strike=0,
            entry_price=round(pos["entry_price"], 2),
            exit_price=round(exit_price, 2),
            underlying_entry=round(pos["entry_price"], 2),
            underlying_exit=round(exit_price, 2),
            underlying_move_pct=round(
                (exit_price - pos["entry_price"]) / pos["entry_price"] * 100, 2,
            ),
            pnl_per_contract=round(net_pnl / shares, 2),
            pnl_pct=round(pnl_pct, 2),
            outcome=TradeOutcome.WIN if pnl_pct > 0.5 else TradeOutcome.LOSS,
            signal_score=9.0, signal_type=mode,
            hold_days=pos["days_held"], strategy=mode,
            regime="uptrend", conviction=9.0,
            exit_reason="end_of_backtest",
            contracts=shares,
            cost_basis=round(pos["cost_basis"], 2),
            exit_value=round(pos["cost_basis"] + net_pnl, 2),
        ))

    # Compile
    trades = closed
    total = len(trades)
    wins = sum(1 for t in trades if t.outcome == TradeOutcome.WIN)
    win_pnls = [t.pnl_pct for t in trades if t.outcome == TradeOutcome.WIN]
    loss_pnls = [t.pnl_pct for t in trades if t.outcome != TradeOutcome.WIN]
    gp = sum(t.pnl_pct for t in trades if t.pnl_pct > 0)
    gl = abs(sum(t.pnl_pct for t in trades if t.pnl_pct < 0))

    by_ticker: dict[str, dict[str, float]] = {}
    for tk in tickers:
        tt = [t for t in trades if t.ticker == tk]
        if tt:
            tw = sum(1 for t in tt if t.outcome == TradeOutcome.WIN)
            by_ticker[tk] = {
                "trades": float(len(tt)),
                "win_rate": round(tw / len(tt), 3),
                "avg_pnl_pct": round(sum(t.pnl_pct for t in tt) / len(tt), 2),
                "total_pnl_pct": round(sum(t.pnl_pct for t in tt), 2),
            }

    ending = daily_values[-1][1] if daily_values else starting_capital
    ret = (ending - starting_capital) / starting_capital * 100

    peak = starting_capital
    max_dd = 0.0
    for _, v in daily_values:
        if v > peak:
            peak = v
        max_dd = max(max_dd, (peak - v) / peak * 100 if peak > 0 else 0)

    sharpe = 0.0
    if len(daily_values) > 10:
        rets = [
            (daily_values[i][1] - daily_values[i - 1][1]) / daily_values[i - 1][1]
            for i in range(1, len(daily_values)) if daily_values[i - 1][1] > 0
        ]
        if rets:
            m = sum(rets) / len(rets)
            var = sum((r - m) ** 2 for r in rets) / len(rets)
            s = sqrt(var) if var > 0 else 0.001
            sharpe = round((m * 252 - 0.05) / (s * sqrt(252)), 3)

    result = BacktestResult(
        run_id=f"{mode}-{uuid.uuid4().hex[:6]}",
        tickers=tickers, lookback_days=730,
        total_trades=total, wins=wins, losses=total - wins,
        win_rate=round(wins / total, 3) if total > 0 else 0,
        avg_win_pct=round(sum(win_pnls) / len(win_pnls), 2) if win_pnls else 0,
        avg_loss_pct=round(sum(loss_pnls) / len(loss_pnls), 2) if loss_pnls else 0,
        best_trade_pct=round(max((t.pnl_pct for t in trades), default=0), 2),
        worst_trade_pct=round(min((t.pnl_pct for t in trades), default=0), 2),
        total_pnl_pct=round(sum(t.pnl_pct for t in trades), 2),
        profit_factor=round(gp / gl, 2) if gl > 0 else 0,
        avg_hold_days=round(sum(t.hold_days for t in trades) / total, 1) if total > 0 else 0,
        expectancy_pct=round(sum(t.pnl_pct for t in trades) / total, 2) if total > 0 else 0,
        trades=trades, by_ticker=by_ticker,
        starting_capital=starting_capital,
        ending_value=round(ending, 2),
        portfolio_return_pct=round(ret, 2),
        max_drawdown_pct=round(max_dd, 2),
        sharpe_ratio=sharpe,
    )

    if total >= 5 and _record:
        from flowedge.scanner.backtest.learning_hook import post_backtest_learn_from_result
        post_backtest_learn_from_result(result, model_name=mode)

    if _record:
        from flowedge.scanner.backtest.run_history import record_run
        record_run(result, model_name=mode, params=overrides or None)

    return result
