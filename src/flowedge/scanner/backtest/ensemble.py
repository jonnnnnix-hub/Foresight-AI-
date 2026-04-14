"""Ensemble portfolio manager — combines specialist bots into a portfolio.

Each specialist bot independently scans its tickers. The portfolio manager:
1. Collects all signals from all specialists
2. Ranks by P(win) from the trained scorer (Phase 4)
3. Allocates capital based on specialist track record + risk constraints
4. Limits total portfolio heat (max concurrent positions)
5. Checks sector correlation (don't stack 3 semiconductor names)

The ensemble backtest runs all specialists together through time,
simulating capital allocation across bots.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import date
from math import sqrt
from pathlib import Path
from typing import Any

import structlog

from flowedge.scanner.backtest.schemas import (
    BacktestResult,
    BacktestTrade,
    TradeOutcome,
)
from flowedge.scanner.backtest.shares_engine import (
    REBATE_PER_SHARE,
    _load_daily,
    _rsi,
)
from flowedge.scanner.backtest.specialist import (
    SpecialistConfig,
    load_specialists,
)

logger = structlog.get_logger()


# Sector groupings to limit correlation
SECTOR_MAP: dict[str, str] = {
    "AAPL": "tech", "MSFT": "tech", "GOOGL": "tech", "META": "tech",
    "AMZN": "tech", "NFLX": "tech", "CRM": "tech",
    "NVDA": "semi", "AMD": "semi", "AVGO": "semi", "INTC": "semi",
    "ARM": "semi", "SMCI": "semi",
    "SPY": "index", "QQQ": "index", "IWM": "index", "DIA": "index",
    "XLK": "sector_etf", "XLF": "sector_etf", "XLE": "sector_etf", "XLV": "sector_etf",
    "JPM": "finance", "BAC": "finance", "V": "finance",
    "TSLA": "auto", "COIN": "crypto", "MSTR": "crypto",
    "PLTR": "defense", "SOFI": "fintech", "HOOD": "fintech",
    "COST": "retail", "WMT": "retail",
    "RDDT": "social",
}


@dataclass
class EnsembleConfig:
    """Configuration for the ensemble portfolio manager."""

    max_positions: int = 4
    max_per_sector: int = 2
    starting_capital: float = 10_000.0
    min_conviction: float = 5.5  # P(win) × 10 minimum
    rebalance_on_close: bool = True


@dataclass
class BotSignal:
    """A signal from a specialist bot on a given day."""

    ticker: str
    specialist_name: str
    params: dict[str, float]
    conviction: float  # 0-10 from scorer
    sector: str
    entry_price: float
    day_idx: int
    date_str: str


@dataclass
class EnsembleResult:
    """Results from the ensemble backtest."""

    config: EnsembleConfig
    specialists_used: list[str]
    backtest: BacktestResult
    per_specialist: dict[str, dict[str, float]] = field(default_factory=dict)
    per_sector: dict[str, dict[str, float]] = field(default_factory=dict)


# ── Ensemble Backtest Engine ─────────────────────────────────────────────────


def run_ensemble_backtest(
    specialists: list[SpecialistConfig] | None = None,
    config: EnsembleConfig | None = None,
) -> EnsembleResult:
    """Run ensemble backtest combining all specialist bots.

    This runs through time day-by-day, letting each specialist generate
    signals with its own optimized parameters. The portfolio manager then
    selects the best signals subject to position limits and sector constraints.
    """
    if specialists is None:
        specialists = load_specialists()
    if not specialists:
        raise ValueError("No specialist configs available")

    cfg = config or EnsembleConfig()
    starting_capital = cfg.starting_capital

    # Only use shares specialists for now (simpler, no options data sync needed)
    shares_specs = [s for s in specialists if s.instrument in ("shares", "both")]
    if not shares_specs:
        raise ValueError("No shares specialists available")

    # Collect all tickers across specialists
    all_tickers: set[str] = set()
    ticker_to_spec: dict[str, SpecialistConfig] = {}
    for spec in shares_specs:
        for ticker in spec.tickers:
            all_tickers.add(ticker)
            ticker_to_spec[ticker] = spec

    # Load daily data for all tickers
    all_daily: dict[str, list[dict[str, Any]]] = {}
    for ticker in all_tickers:
        daily = _load_daily(ticker)
        if len(daily) >= 60:
            all_daily[ticker] = daily

    if not all_daily:
        raise ValueError("No daily data loaded")

    # Get all trading dates
    all_dates: set[str] = set()
    for bars in all_daily.values():
        for b in bars:
            all_dates.add(b["date"])
    sorted_dates = sorted(all_dates)

    # State
    cash = starting_capital
    positions: list[dict[str, Any]] = []
    closed: list[BacktestTrade] = []
    daily_values: list[tuple[str, float]] = []
    last_entry: dict[str, int] = {}
    specialist_trades: dict[str, list[BacktestTrade]] = {
        s.name: [] for s in shares_specs
    }

    for day_idx, d in enumerate(sorted_dates):
        # ── Process exits ──
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

            params = pos["params"]
            tp_pct = params.get("tp_pct", 0.025)
            max_hold = int(params.get("max_hold", 12))
            min_hold = int(params.get("min_hold", 3))
            trail_trigger = params.get("trail_trigger", 0.015)
            trail_exit = params.get("trail_exit", 0.005)

            if gain >= tp_pct:
                to_close.append((pos, "take_profit"))
                continue

            if pos["days_held"] < min_hold:
                continue

            if pos["days_held"] >= max_hold:
                to_close.append((pos, "time_exit"))
                continue

            if max_gain >= trail_trigger and gain < trail_exit:
                to_close.append((pos, "trailing_stop"))
                continue

        for pos, reason in to_close:
            if pos not in positions:
                continue
            shares = pos["shares"]
            exit_price = pos["current_price"]
            pnl = (exit_price - pos["entry_price"]) * shares
            rebate = shares * REBATE_PER_SHARE * 2
            net_pnl = pnl + rebate
            pnl_pct = net_pnl / pos["cost_basis"] * 100

            cash += pos["cost_basis"] + net_pnl
            positions.remove(pos)

            outcome = TradeOutcome.WIN if pnl_pct > 0.5 else (
                TradeOutcome.LOSS if pnl_pct < -0.5 else TradeOutcome.BREAKEVEN
            )

            trade = BacktestTrade(
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
                signal_score=pos.get("conviction", 9.0),
                signal_type=f"ensemble|{pos['specialist']}|{shares}sh",
                hold_days=pos["days_held"],
                strategy=pos["specialist"],
                regime="uptrend",
                conviction=pos.get("conviction", 9.0),
                exit_reason=reason,
                contracts=shares,
                cost_basis=round(pos["cost_basis"], 2),
                exit_value=round(pos["cost_basis"] + net_pnl, 2),
            )
            closed.append(trade)
            if pos["specialist"] in specialist_trades:
                specialist_trades[pos["specialist"]].append(trade)

        # ── Generate signals from all specialists ──
        if len(positions) >= cfg.max_positions:
            pos_value = sum(p["current_price"] * p["shares"] for p in positions)
            daily_values.append((d, cash + pos_value))
            continue

        signals: list[BotSignal] = []
        for ticker, spec in ticker_to_spec.items():
            if any(p["ticker"] == ticker for p in positions):
                continue
            if day_idx - last_entry.get(ticker, -999) < 3:
                continue

            tk_bars = all_daily.get(ticker, [])
            idx = next(
                (i for i, b in enumerate(tk_bars) if b["date"] == d), None,
            )
            if idx is None or idx < 55:
                continue

            bar = tk_bars[idx]
            closes = [tk_bars[j]["close"] for j in range(max(0, idx - 49), idx + 1)]
            if len(closes) < 50:
                continue

            params = spec.shares_params
            ibs_thresh = params.get("ibs_threshold", 0.20)
            rsi_thresh = params.get("rsi_threshold", 45.0)

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

            # Compute conviction from specialist's historical WR
            conviction = spec.optimized_win_rate * 10

            signals.append(BotSignal(
                ticker=ticker,
                specialist_name=spec.name,
                params=params,
                conviction=conviction,
                sector=SECTOR_MAP.get(ticker, "other"),
                entry_price=bar["close"],
                day_idx=day_idx,
                date_str=d,
            ))

        # ── Portfolio manager: rank and filter ──
        signals.sort(key=lambda s: s.conviction, reverse=True)

        # Track sector exposure
        current_sectors: dict[str, int] = {}
        for pos in positions:
            sector = SECTOR_MAP.get(pos["ticker"], "other")
            current_sectors[sector] = current_sectors.get(sector, 0) + 1

        for sig in signals:
            if len(positions) >= cfg.max_positions:
                break

            if sig.conviction < cfg.min_conviction:
                continue

            # Sector limit
            if current_sectors.get(sig.sector, 0) >= cfg.max_per_sector:
                continue

            # Enter position
            risk_pct = sig.params.get("risk_pct", 0.10)
            budget = cash * risk_pct
            shares = int(budget / sig.entry_price)
            if shares < 1:
                continue
            cost = shares * sig.entry_price
            if cost > cash * 0.95:
                continue

            cash -= cost
            positions.append({
                "ticker": sig.ticker,
                "entry_date": d,
                "entry_price": sig.entry_price,
                "shares": shares,
                "cost_basis": cost,
                "days_held": 0,
                "current_price": sig.entry_price,
                "max_price": sig.entry_price,
                "specialist": sig.specialist_name,
                "params": sig.params,
                "conviction": sig.conviction,
            })
            last_entry[sig.ticker] = day_idx
            current_sectors[sig.sector] = current_sectors.get(sig.sector, 0) + 1

        # Daily snapshot
        pos_value = sum(p["current_price"] * p["shares"] for p in positions)
        daily_values.append((d, cash + pos_value))

    # Close remaining positions
    for pos in positions[:]:
        exit_price = pos["current_price"]
        shares = pos["shares"]
        pnl = (exit_price - pos["entry_price"]) * shares
        rebate = shares * REBATE_PER_SHARE * 2
        net_pnl = pnl + rebate
        pnl_pct = net_pnl / pos["cost_basis"] * 100
        cash += pos["cost_basis"] + net_pnl

        trade = BacktestTrade(
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
            signal_score=pos.get("conviction", 9.0),
            signal_type=f"ensemble|{pos['specialist']}",
            hold_days=pos["days_held"], strategy=pos["specialist"],
            regime="uptrend", conviction=pos.get("conviction", 9.0),
            exit_reason="end_of_backtest",
            contracts=shares,
            cost_basis=round(pos["cost_basis"], 2),
            exit_value=round(pos["cost_basis"] + net_pnl, 2),
        )
        closed.append(trade)
        if pos["specialist"] in specialist_trades:
            specialist_trades[pos["specialist"]].append(trade)

    # Compile results
    trades = closed
    total = len(trades)
    wins = sum(1 for t in trades if t.outcome == TradeOutcome.WIN)
    win_pnls = [t.pnl_pct for t in trades if t.outcome == TradeOutcome.WIN]
    loss_pnls = [t.pnl_pct for t in trades if t.outcome != TradeOutcome.WIN]
    gp = sum(t.pnl_pct for t in trades if t.pnl_pct > 0)
    gl = abs(sum(t.pnl_pct for t in trades if t.pnl_pct < 0))

    by_ticker: dict[str, dict[str, float]] = {}
    for tk in all_tickers:
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

    backtest_result = BacktestResult(
        run_id=f"ensemble-{uuid.uuid4().hex[:8]}",
        tickers=sorted(all_tickers),
        lookback_days=730,
        total_trades=total,
        wins=wins,
        losses=total - wins,
        win_rate=round(wins / total, 3) if total > 0 else 0,
        avg_win_pct=round(sum(win_pnls) / len(win_pnls), 2) if win_pnls else 0,
        avg_loss_pct=round(sum(loss_pnls) / len(loss_pnls), 2) if loss_pnls else 0,
        best_trade_pct=round(max((t.pnl_pct for t in trades), default=0), 2),
        worst_trade_pct=round(min((t.pnl_pct for t in trades), default=0), 2),
        total_pnl_pct=round(sum(t.pnl_pct for t in trades), 2),
        profit_factor=round(gp / gl, 2) if gl > 0 else 0,
        avg_hold_days=round(
            sum(t.hold_days for t in trades) / total, 1,
        ) if total > 0 else 0,
        expectancy_pct=round(
            sum(t.pnl_pct for t in trades) / total, 2,
        ) if total > 0 else 0,
        trades=trades,
        by_ticker=by_ticker,
        starting_capital=starting_capital,
        ending_value=round(ending, 2),
        portfolio_return_pct=round(ret, 2),
        max_drawdown_pct=round(max_dd, 2),
        sharpe_ratio=sharpe,
    )

    # Per-specialist breakdown
    per_spec: dict[str, dict[str, float]] = {}
    for name, spec_trades in specialist_trades.items():
        if spec_trades:
            sw = sum(1 for t in spec_trades if t.outcome == TradeOutcome.WIN)
            per_spec[name] = {
                "trades": float(len(spec_trades)),
                "wins": float(sw),
                "win_rate": round(sw / len(spec_trades), 3),
                "total_pnl_pct": round(sum(t.pnl_pct for t in spec_trades), 2),
            }

    ensemble_result = EnsembleResult(
        config=cfg,
        specialists_used=[s.name for s in shares_specs],
        backtest=backtest_result,
        per_specialist=per_spec,
    )

    _print_ensemble_report(ensemble_result)

    # Save
    output_path = Path("data/backtest/ensemble_result.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(
        backtest_result.model_dump(mode="json"),
        indent=2,
        default=str,
    ))

    return ensemble_result


# ── Reporting ────────────────────────────────────────────────────────────────


def _print_ensemble_report(result: EnsembleResult) -> None:
    """Print ensemble backtest results."""
    bt = result.backtest
    print("\n" + "=" * 90)
    print("ENSEMBLE PORTFOLIO — COMBINED SPECIALIST BOTS")
    print("=" * 90)
    print(f"  Specialists: {len(result.specialists_used)}")
    print(f"  Tickers:     {len(bt.tickers)}")
    print(f"  Max Positions: {result.config.max_positions}")
    print(f"  Capital:     ${bt.starting_capital:,.0f}")
    print()
    print(f"  Total Trades:  {bt.total_trades}")
    print(f"  Win Rate:      {bt.win_rate:.1%}")
    print(f"  Return:        {bt.portfolio_return_pct:+.1f}%")
    print(f"  Profit Factor: {bt.profit_factor:.2f}")
    print(f"  Max Drawdown:  {bt.max_drawdown_pct:.1f}%")
    print(f"  Sharpe Ratio:  {bt.sharpe_ratio:.3f}")
    print(f"  Avg Hold Days: {bt.avg_hold_days:.1f}")
    print()

    if result.per_specialist:
        print("  PER-SPECIALIST BREAKDOWN:")
        print(f"  {'Specialist':<25} {'Trades':>7} {'WR':>7} {'PnL%':>8}")
        print("  " + "-" * 50)
        for name, stats in sorted(
            result.per_specialist.items(),
            key=lambda x: x[1].get("win_rate", 0),
            reverse=True,
        ):
            print(
                f"  {name:<25} "
                f"{stats['trades']:>7.0f} "
                f"{stats['win_rate']:>6.1%} "
                f"{stats['total_pnl_pct']:>+7.1f}%"
            )

    print()
    if bt.by_ticker:
        print("  TOP TICKERS:")
        print(f"  {'Ticker':<8} {'Trades':>7} {'WR':>7} {'PnL%':>8}")
        print("  " + "-" * 35)
        sorted_tickers = sorted(
            bt.by_ticker.items(),
            key=lambda x: x[1].get("win_rate", 0),
            reverse=True,
        )[:10]
        for tk, stats in sorted_tickers:
            print(
                f"  {tk:<8} "
                f"{stats['trades']:>7.0f} "
                f"{stats['win_rate']:>6.1%} "
                f"{stats['total_pnl_pct']:>+7.1f}%"
            )

    print("=" * 90 + "\n")


# ── CLI ──────────────────────────────────────────────────────────────────────


if __name__ == "__main__":
    from flowedge.config.logging import setup_logging

    setup_logging("INFO")
    run_ensemble_backtest()
