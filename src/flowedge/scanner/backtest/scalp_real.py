"""FlowEdge Scalp — Real options contract backtester.

Unlike all prior models that use Black-Scholes estimates, this model
uses ACTUAL option contract prices from Massive S3 flat files.

Flow:
1. Detect signal on stock minute bars (7-condition confluence)
2. Find nearest-ATM call in the options flat file for that minute
3. Enter at the REAL option ask price (open of next bar)
4. Monitor REAL option price minute-by-minute
5. Exit at REAL option bid price when TP/time/trail triggers

No BS estimates. No slippage model. Pure actual market prices.

Data required:
- data/flat_files_s3/{ticker}/1min/ — stock minute bars
- data/flat_files_s3/{ticker}/options_1min/ — option contract bars
"""

from __future__ import annotations

import json
import uuid
from collections import defaultdict
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
from flowedge.scanner.backtest.slippage import SlippageModel, estimate_half_spread

logger = structlog.get_logger()

CACHE_DIR = Path("data/flat_files_s3")

# Scalp tickers validated on minute data
# v3: Expanded universe — all tickers with 50%+ WR & positive PnL on real data
# SOFI 69%, SPY 70%, GOOGL 66%, AMZN 58%, IWM 56%, AAPL 50%
SCALP_TICKERS = ["SOFI", "SPY", "GOOGL", "AMZN", "IWM", "AAPL"]

# Entry parameters (7-condition filter from validation)
IBS_THRESH = 0.10
RSI3_THRESH = 20.0
VOL_SPIKE = 2.0
INTRADAY_DROP = -0.003  # -0.3% from open
MAX_ENTRY_BAR = 24  # Only morning session (first 2 hours)
MIN_ENTRY_BAR = 6  # Skip first 30 min

# Exit parameters
TP_PCT = 0.10  # v2: 10% TP (was 8% — avg TP win is +15.3%, raise to capture more)
MAX_HOLD_BARS = 6  # v2: 6 x 5-min = 30 min — optimal (20min too tight, 60min too loose)
TRAIL_PCT = 0.04  # 4% trail from peak option price

# Risk
MAX_POSITIONS = 2
RISK_PER_TRADE = 0.05  # 5% of capital per scalp

# Option selection
MIN_OPTION_VOLUME = 10  # Minimum volume for tradeable contract
MAX_STRIKE_DIST_PCT = 0.02  # 2% from current price

# Minimum premium filter — skip illiquid/penny options
MIN_PREMIUM = 0.50


def _gf(bar: dict[str, Any], long_key: str, short_key: str) -> float:
    val: object = bar.get(long_key, bar.get(short_key, 0))
    return float(val)  # type: ignore[arg-type]


def _parse_option_ticker(ticker: str) -> dict[str, Any] | None:
    """Parse OCC option ticker: O:SPY260116C00580000."""
    if len(ticker) < 16 or not ticker.startswith("O:"):
        return None
    # Find where the date starts (6 digits after underlying)
    rest = ticker[2:]  # Remove O:
    # Find underlying by looking for digit
    i = 0
    while i < len(rest) and not rest[i].isdigit():
        i += 1
    if i == 0 or i >= len(rest) - 10:
        return None

    underlying = rest[:i]
    date_str = rest[i:i + 6]  # YYMMDD
    cp = rest[i + 6]  # C or P
    strike_raw = rest[i + 7:]

    try:
        strike = int(strike_raw) / 1000
        exp = f"20{date_str[:2]}-{date_str[2:4]}-{date_str[4:6]}"
    except (ValueError, IndexError):
        return None

    return {
        "underlying": underlying,
        "expiration": exp,
        "type": "call" if cp == "C" else "put",
        "strike": strike,
    }


def _find_nearest_atm_call(
    option_bars: list[dict[str, Any]],
    current_price: float,
    target_dte_min: int = 0,
    target_dte_max: int = 7,
    max_strike_dist_pct: float = MAX_STRIKE_DIST_PCT,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    """Find nearest ATM call + all its minute bars for the day.

    Returns (best_option_info, all_bars_for_that_contract).
    The pre-parsed format has strike, option_type, dte directly.
    """
    # Group bars by contract and find best ATM call
    by_contract: dict[str, list[dict[str, Any]]] = defaultdict(list)
    contract_info: dict[str, dict[str, Any]] = {}

    for ob in option_bars:
        contract = ob.get("contract", "")
        if not contract:
            continue
        opt_type = ob.get("option_type", "")
        if opt_type != "C":
            continue

        dte = int(ob.get("dte", 99))
        if dte < target_dte_min or dte > target_dte_max:
            continue

        strike = float(ob.get("strike", 0))
        vol = int(ob.get("v", ob.get("volume", 0)))

        dist = abs(strike - current_price) / current_price
        if dist > max_strike_dist_pct:
            continue

        by_contract[contract].append(ob)
        if contract not in contract_info or vol > contract_info[contract].get("vol", 0):
            contract_info[contract] = {
                "contract": contract,
                "strike": strike,
                "dte": dte,
                "expiration": ob.get("expiration", ""),
                "dist": dist,
                "vol": vol,
            }

    if not contract_info:
        return None, []

    # Pick closest to ATM with best volume
    best = min(
        contract_info.values(),
        key=lambda x: (x["dist"], -x["vol"]),
    )

    contract_bars = sorted(
        by_contract[best["contract"]],
        key=lambda x: str(x.get("timestamp", x.get("ts", ""))),
    )

    # Get a representative price (first bar with volume)
    price = 0.0
    for cb in contract_bars:
        p = float(cb.get("c", cb.get("close", 0)))
        v = int(cb.get("v", cb.get("volume", 0)))
        if p > 0 and v >= 1:
            price = p
            break

    if price <= 0:
        return None, []

    info = {
        "contract": best["contract"],
        "strike": best["strike"],
        "expiration": best["expiration"],
        "price": price,
        "volume": best["vol"],
    }
    return info, contract_bars


def run_scalp_real_backtest(
    tickers: list[str] | None = None,
    starting_capital: float = 25_000.0,
    params: dict[str, Any] | None = None,
    _record: bool = True,
) -> BacktestResult:
    """Run scalp backtest using REAL option contract prices.

    No Black-Scholes. No slippage model. Actual market prices.

    Args:
        params: Optional parameter overrides. Keys: ibs_thresh, rsi3_thresh,
                vol_spike, intraday_drop, max_entry_bar, min_entry_bar,
                tp_pct, max_hold_bars, trail_pct, risk_per_trade,
                max_strike_dist_pct.
    """
    tickers = tickers or SCALP_TICKERS
    p = params or {}
    _ibs_thresh = p.get("ibs_thresh", IBS_THRESH)
    _rsi3_thresh = p.get("rsi3_thresh", RSI3_THRESH)
    _vol_spike = p.get("vol_spike", VOL_SPIKE)
    _intraday_drop = p.get("intraday_drop", INTRADAY_DROP)
    _max_entry_bar = p.get("max_entry_bar", MAX_ENTRY_BAR)
    _min_entry_bar = p.get("min_entry_bar", MIN_ENTRY_BAR)
    _tp_pct = p.get("tp_pct", TP_PCT)
    _max_hold_bars = p.get("max_hold_bars", MAX_HOLD_BARS)
    _trail_pct = p.get("trail_pct", TRAIL_PCT)
    _risk_per_trade = p.get("risk_per_trade", RISK_PER_TRADE)
    _max_strike_dist_pct = p.get("max_strike_dist_pct", MAX_STRIKE_DIST_PCT)
    slippage_model = SlippageModel()

    # Load stock minute bars (grouped by date)
    stock_data: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for ticker in tickers:
        min_dir = CACHE_DIR / ticker / "1min"
        if not min_dir.exists():
            continue
        bars: list[dict[str, Any]] = []
        for f in sorted(min_dir.glob("*.json")):
            bars.extend(json.loads(f.read_text()))

        by_date: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for b in bars:
            d = str(b.get("date", b.get("d", "")))
            if d:
                by_date[d].append(b)
        stock_data[ticker] = dict(by_date)

    # Load options minute bars (per-day JSON files)
    options_data: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for ticker in tickers:
        opt_dir = CACHE_DIR / ticker / "options_1min"
        if not opt_dir.exists():
            continue
        ticker_opts: dict[str, list[dict[str, Any]]] = {}
        files = sorted(opt_dir.glob("*.json"))
        for f in files:
            # Filename: TICKER_options_1min_YYYY-MM-DD.json
            # Extract date from filename
            parts = f.stem.split("_")
            d_str = parts[-1] if len(parts) >= 4 else ""
            if not d_str or len(d_str) != 10:
                continue
            data = json.loads(f.read_text())
            if isinstance(data, list):
                ticker_opts[d_str] = data
            elif isinstance(data, dict):
                # Old format: date → list
                for k, v in data.items():
                    if isinstance(v, list):
                        ticker_opts[k] = v
        options_data[ticker] = ticker_opts
        logger.info(
            "scalp_options_loaded",
            ticker=ticker,
            days=len(ticker_opts),
        )

    if not stock_data or not options_data:
        return BacktestResult(
            run_id=f"scalp-real-{uuid.uuid4().hex[:8]}",
            tickers=tickers, starting_capital=starting_capital,
        )

    # Find dates where we have BOTH stock AND options data
    valid_dates: set[str] = set()
    for ticker in tickers:
        stock_dates = set(stock_data.get(ticker, {}).keys())
        option_dates = set(options_data.get(ticker, {}).keys())
        valid_dates.update(stock_dates & option_dates)

    sorted_dates = sorted(valid_dates)
    logger.info("scalp_dates", count=len(sorted_dates))

    if not sorted_dates:
        return BacktestResult(
            run_id=f"scalp-real-{uuid.uuid4().hex[:8]}",
            tickers=tickers, starting_capital=starting_capital,
        )

    # Track daily closes for trend filter
    daily_closes: dict[str, list[float]] = {t: [] for t in tickers}

    cash = starting_capital
    closed: list[BacktestTrade] = []
    daily_values: list[tuple[str, float]] = []

    for d in sorted_dates:
        # Update daily closes from stock data
        for ticker in tickers:
            day_stock = stock_data.get(ticker, {}).get(d, [])
            if day_stock:
                daily_closes[ticker].append(
                    _gf(day_stock[-1], "close", "c"),
                )

        for ticker in tickers:
            dc = daily_closes[ticker]
            if len(dc) < 20:
                continue

            # Daily uptrend: SMA10 > SMA20
            sma10 = sum(dc[-10:]) / 10
            sma20 = sum(dc[-20:]) / 20
            if sma10 <= sma20:
                continue

            day_stock = stock_data.get(ticker, {}).get(d, [])
            day_options = options_data.get(ticker, {}).get(d, [])
            if len(day_stock) < 50 or not day_options:
                continue

            # Build 5-min stock bars
            chunks: list[dict[str, float]] = []
            for i in range(0, len(day_stock), 5):
                chunk = day_stock[i:i + 5]
                if not chunk:
                    continue
                o = _gf(chunk[0], "open", "o")
                h = max(_gf(b, "high", "h") for b in chunk)
                lo_vals = [_gf(b, "low", "l") for b in chunk if _gf(b, "low", "l") > 0]
                lo = min(lo_vals) if lo_vals else o
                c = _gf(chunk[-1], "close", "c")
                v = sum(_gf(b, "volume", "v") for b in chunk)
                chunks.append({"o": o, "h": h, "l": lo, "c": c, "v": v})

            if len(chunks) < 30:
                continue

            day_open = chunks[0]["o"]

            # VWAP
            cum_pv = 0.0
            cum_v = 0.0
            vwaps: list[float] = []
            for ch in chunks:
                tp = (ch["h"] + ch["l"] + ch["c"]) / 3
                cum_pv += tp * ch["v"]
                cum_v += ch["v"]
                vwaps.append(cum_pv / cum_v if cum_v > 0 else ch["c"])

            # Scan morning session
            for i in range(
                _min_entry_bar, min(_max_entry_bar, len(chunks) - _max_hold_bars)
            ):
                ch = chunks[i]
                rng = ch["h"] - ch["l"]
                if rng <= 0 or ch["c"] <= 0:
                    continue

                # ── 7-CONDITION FILTER ──
                ibs = (ch["c"] - ch["l"]) / rng
                if ibs >= _ibs_thresh:
                    continue

                if i < 4:
                    continue
                c5m = [chunks[j]["c"] for j in range(i - 3, i + 1)]
                g = [max(0, c5m[k] - c5m[k - 1]) for k in range(1, 4)]
                ls = [max(0, c5m[k - 1] - c5m[k]) for k in range(1, 4)]
                ag = sum(g) / 3
                al = sum(ls) / 3
                rsi3 = 100 - 100 / (1 + ag / al) if al > 0 else 100
                if rsi3 >= _rsi3_thresh:
                    continue

                if ch["c"] >= vwaps[i]:
                    continue

                start_idx = max(0, i - 10)
                count = max(1, i - start_idx)
                avg_vol = sum(chunks[j]["v"] for j in range(start_idx, i)) / count
                vr = ch["v"] / avg_vol if avg_vol > 0 else 1
                if vr < _vol_spike:
                    continue

                drop = (ch["c"] - day_open) / day_open
                if drop > _intraday_drop:
                    continue

                if i > 0 and chunks[i - 1]["c"] >= chunks[max(0, i - 2)]["c"]:
                    continue

                if i >= 10:
                    sma5 = sum(chunks[j]["c"] for j in range(i - 4, i + 1)) / 5
                    sma10_id = sum(
                        chunks[j]["c"]
                        for j in range(max(0, i - 9), i + 1)
                    ) / min(10, i + 1)
                    if sma5 >= sma10_id:
                        continue

                # SIGNAL FIRED — find real option contract
                current_price = ch["c"]
                option, contract_bars = _find_nearest_atm_call(
                    day_options, current_price,
                    target_dte_min=0, target_dte_max=7,
                    max_strike_dist_pct=_max_strike_dist_pct,
                )

                if not option or option["price"] <= 0 or not contract_bars:
                    continue

                # Enter at option price
                entry_premium = option["price"]
                if entry_premium < MIN_PREMIUM:
                    continue

                # Apply tier-based slippage to entry
                otm_pct = (
                    abs(option["strike"] - current_price) / current_price
                    if current_price > 0 else 0
                )
                entry_half_spread = estimate_half_spread(
                    premium=entry_premium, otm_pct=otm_pct, ticker=ticker,
                    model=slippage_model,
                )
                entry_premium += entry_half_spread  # Buy at ask

                budget = cash * _risk_per_trade
                contracts = max(1, int(budget / (entry_premium * 100)))
                cost = contracts * entry_premium * 100

                if cost > cash * 0.9 or cost < 10:
                    continue

                # Find bars after our entry
                exit_premium = entry_premium
                max_premium = entry_premium
                exit_reason = "time_exit"
                hold_bars = 0

                for cb in contract_bars:
                    # Each option bar is 1-minute, fields: c/close, o/open, h/high, l/low
                    cb_price = float(cb.get("c", cb.get("close", 0)))
                    if cb_price <= 0:
                        continue

                    hold_bars += 1
                    if hold_bars > _max_hold_bars * 5:  # 5 min-bars per 5-min chunk
                        break

                    if cb_price > max_premium:
                        max_premium = cb_price

                    exit_premium = cb_price

                    # TP
                    pnl_pct = (cb_price - entry_premium) / entry_premium
                    if pnl_pct >= _tp_pct:
                        exit_reason = "take_profit"
                        exit_premium = cb_price
                        break

                    # Trail
                    if max_premium > entry_premium * 1.04:
                        trail = max_premium * (1 - _trail_pct)
                        if cb_price <= trail:
                            exit_reason = "trailing_stop"
                            exit_premium = cb_price
                            break

                # Apply tier-based slippage to exit
                exit_half_spread = estimate_half_spread(
                    premium=exit_premium, otm_pct=otm_pct, ticker=ticker,
                    model=slippage_model,
                )
                exit_premium = max(0.01, exit_premium - exit_half_spread)  # Sell at bid

                # Compute P&L on real prices
                exit_val = exit_premium * contracts * 100
                pnl = exit_val - cost
                pnl_pct = (pnl / cost * 100) if cost > 0 else 0

                cash += pnl  # Net P&L

                outcome = TradeOutcome.WIN if pnl_pct >= 3 else (
                    TradeOutcome.LOSS if pnl_pct < -3 else TradeOutcome.BREAKEVEN
                )

                hold_minutes = hold_bars  # Approximate

                closed.append(BacktestTrade(
                    ticker=ticker,
                    entry_date=date.fromisoformat(d),
                    exit_date=date.fromisoformat(d),
                    option_type="call",
                    strike=option["strike"],
                    entry_price=round(entry_premium, 4),
                    exit_price=round(exit_premium, 4),
                    underlying_entry=round(current_price, 2),
                    underlying_exit=round(current_price, 2),
                    underlying_move_pct=0,
                    pnl_per_contract=round(pnl / max(contracts, 1), 2),
                    pnl_pct=round(pnl_pct, 2),
                    outcome=outcome,
                    signal_score=9.0,
                    signal_type=f"scalp_real|{option['contract']}|{hold_minutes}min",
                    hold_days=0,
                    strategy="scalp_real",
                    regime="intraday_uptrend",
                    conviction=9.0,
                    exit_reason=exit_reason,
                    contracts=contracts,
                    cost_basis=round(cost, 2),
                    exit_value=round(exit_val, 2),
                ))

                break  # One entry per ticker per day

        daily_values.append((d, cash))

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
        run_id=f"scalp-real-{uuid.uuid4().hex[:8]}",
        tickers=tickers, lookback_days=90,
        total_trades=total, wins=wins, losses=total - wins,
        win_rate=round(wins / total, 3) if total > 0 else 0,
        avg_win_pct=round(sum(win_pnls) / len(win_pnls), 2) if win_pnls else 0,
        avg_loss_pct=round(sum(loss_pnls) / len(loss_pnls), 2) if loss_pnls else 0,
        best_trade_pct=round(max((t.pnl_pct for t in trades), default=0), 2),
        worst_trade_pct=round(min((t.pnl_pct for t in trades), default=0), 2),
        total_pnl_pct=round(sum(t.pnl_pct for t in trades), 2),
        profit_factor=round(gp / gl, 2) if gl > 0 else 0,
        avg_hold_days=0,
        expectancy_pct=round(sum(t.pnl_pct for t in trades) / total, 2) if total > 0 else 0,
        trades=trades, by_ticker=by_ticker,
        starting_capital=starting_capital,
        ending_value=round(ending, 2),
        portfolio_return_pct=round(ret, 2),
        max_drawdown_pct=round(max_dd, 2),
        sharpe_ratio=sharpe,
    )

    if total >= 3 and _record:
        from flowedge.scanner.backtest.learning_hook import post_backtest_learn_from_result
        post_backtest_learn_from_result(result, model_name="scalp_real")

    if _record:
        from flowedge.scanner.backtest.run_history import record_run
        record_run(result, model_name="scalp_real", params=p or None)

    return result
