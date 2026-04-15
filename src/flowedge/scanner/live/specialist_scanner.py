"""ZEUS — FlowEdge Specialist Ensemble Scanner — live paper trading.

Runs 77 per-ticker optimized IBS reversion specialists during market hours:
1. Loads specialist configs from data/specialists/ (per-ticker optimized params)
2. Every 5 minutes, pulls daily OHLCV bars from Polygon
3. Checks each specialist's entry conditions (IBS, RSI, SMA, prior-day down)
4. Ranks signals by specialist WR and applies sector/position limits
5. Executes as SHARES via Alpaca paper trading
6. Monitors open positions for per-specialist exits (custom TP, trail, time)

Ensemble backtest: 77.3% WR, +988.9% return, PF 3.66 on 7 years of data.

Usage:
    .venv/bin/python -m flowedge.scanner.live.specialist_scanner
    .venv/bin/python scripts/run_specialist_scanner.py
    .venv/bin/python scripts/run_specialist_scanner.py --dry-run
"""

from __future__ import annotations

import asyncio
import json
import os
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any

import structlog
from dotenv import load_dotenv

from flowedge.scanner.data_feeds.alpaca_execution import (
    AlpacaExecutor,
)
from flowedge.scanner.data_feeds.polygon_intraday import PolygonIntradayProvider
from flowedge.scanner.data_feeds.schemas import Timeframe

logger = structlog.get_logger()

MODEL_NAME = "zeus"

# Timezone
ET_OFFSET = timezone(timedelta(hours=-4))  # EDT

# Market hours
MARKET_OPEN = time(9, 35)
MARKET_CLOSE = time(15, 55)

# Entry window — scan near close to match backtest (entries at close price)
ENTRY_SCAN_TIME = time(15, 50)  # 3:50 PM — 10 min before close

# Intervals
SCAN_INTERVAL = 300  # 5 min signal scan
EXIT_CHECK_INTERVAL = 30  # 30s exit monitoring

# Position limits
MAX_TOTAL_POSITIONS = 4
MAX_PER_SECTOR = 2

# Sector map for correlation control
SECTOR_MAP: dict[str, str] = {
    "AAPL": "tech", "MSFT": "tech", "GOOGL": "tech", "META": "tech",
    "AMZN": "tech", "NFLX": "tech", "CRM": "tech", "ADBE": "tech",
    "NOW": "tech", "ORCL": "tech", "SHOP": "tech", "CRWD": "tech",
    "PANW": "tech",
    "NVDA": "semi", "AMD": "semi", "AVGO": "semi", "INTC": "semi",
    "ARM": "semi", "SMCI": "semi", "MRVL": "semi", "MU": "semi",
    "QCOM": "semi", "TXN": "semi", "LRCX": "semi", "AMAT": "semi",
    "SMH": "semi",
    "SPY": "index", "QQQ": "index", "IWM": "index", "DIA": "index",
    "XLK": "sector_etf", "XLF": "sector_etf", "XLE": "sector_etf",
    "XLV": "sector_etf", "XLI": "sector_etf", "XLC": "sector_etf",
    "XLY": "sector_etf", "XLP": "sector_etf",
    "JPM": "finance", "BAC": "finance", "GS": "finance", "MS": "finance",
    "C": "finance", "WFC": "finance", "V": "finance", "MA": "finance",
    "TSLA": "auto", "COIN": "crypto", "MSTR": "crypto",
    "PLTR": "defense", "SOFI": "fintech", "HOOD": "fintech",
    "PYPL": "fintech", "SQ": "fintech",
    "COST": "retail", "WMT": "retail", "TGT": "retail", "HD": "retail",
    "SBUX": "consumer", "MCD": "consumer", "NKE": "consumer",
    "UNH": "health", "JNJ": "health", "PFE": "health", "LLY": "health",
    "ABBV": "health",
    "XOM": "energy", "CVX": "energy", "COP": "energy",
    "CAT": "industrial", "DE": "industrial", "GE": "industrial",
    "HON": "industrial", "BA": "industrial",
    "GLD": "commodity", "TLT": "bond", "UBER": "rideshare",
}

LOG_DIR = Path("data/live_logs/zeus")


# ── Specialist Config ────────────────────────────────────────────────────────


def _load_specialist_configs() -> list[dict[str, Any]]:
    """Load specialist configs from configs/specialists/ or data/specialists/."""
    configs_dir = Path("configs/specialists")
    if not configs_dir.exists():
        configs_dir = Path("data/specialists")
    if not configs_dir.exists():
        return []

    configs: list[dict[str, Any]] = []
    for f in sorted(configs_dir.glob("*_config.json")):
        data = json.loads(f.read_text())
        if data.get("shares_params"):
            configs.append(data)

    return configs


# ── RSI Calculation ──────────────────────────────────────────────────────────


def _rsi(closes: list[float], period: int = 14) -> float:
    """Compute RSI from a list of close prices."""
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


# ── Signal Detection ────────────────────────────────────────────────────────


def check_specialist_signal(
    daily_bars: list[dict[str, Any]],
    params: dict[str, float],
) -> dict[str, Any] | None:
    """Check if today's bar triggers the specialist's entry signal.

    Requires 55+ days of daily OHLCV bars (sorted ascending by date).
    Returns signal dict or None.
    """
    if len(daily_bars) < 55:
        return None

    bar = daily_bars[-1]  # Today's bar
    prior = daily_bars[-2]  # Yesterday

    ibs_thresh = params.get("ibs_threshold", 0.20)
    rsi_thresh = params.get("rsi_threshold", 45.0)

    closes = [b["close"] for b in daily_bars[-50:]]

    # Uptrend: SMA20 > SMA50
    sma20 = sum(closes[-20:]) / 20
    sma50 = sum(closes) / len(closes)
    if sma20 <= sma50:
        return None

    # IBS filter
    rng = bar["high"] - bar["low"]
    if rng <= 0 or bar["close"] <= 0:
        return None
    ibs = (bar["close"] - bar["low"]) / rng
    if ibs >= ibs_thresh:
        return None

    # RSI filter
    r = _rsi(closes)
    if r >= rsi_thresh:
        return None

    # Prior day down
    if bar["close"] >= prior["close"]:
        return None

    return {
        "ibs": round(ibs, 4),
        "rsi": round(r, 1),
        "sma20": round(sma20, 2),
        "sma50": round(sma50, 2),
        "close": bar["close"],
        "date": bar.get("date", str(date.today())),
    }


# ── Position Tracker ─────────────────────────────────────────────────────────


class PositionTracker:
    """Track open specialist positions and their exit conditions."""

    def __init__(self) -> None:
        self.positions: dict[str, dict[str, Any]] = {}  # ticker → position info
        self.last_entry: dict[str, date] = {}  # ticker → last entry date

    def add(
        self, ticker: str, entry_price: float, qty: int,
        params: dict[str, float], specialist_name: str,
    ) -> None:
        self.positions[ticker] = {
            "ticker": ticker,
            "entry_price": entry_price,
            "qty": qty,
            "entry_date": date.today(),
            "days_held": 0,
            "max_price": entry_price,
            "params": params,
            "specialist": specialist_name,
        }
        self.last_entry[ticker] = date.today()

    def remove(self, ticker: str) -> dict[str, Any] | None:
        return self.positions.pop(ticker, None)

    def can_enter(self, ticker: str, cooldown: int) -> bool:
        if ticker in self.positions:
            return False
        last = self.last_entry.get(ticker)
        return not (last and (date.today() - last).days < cooldown)

    def check_exits(
        self,
        current_prices: dict[str, float],
    ) -> list[tuple[str, str, dict[str, Any]]]:
        """Check exit conditions for all open positions.

        Returns list of (ticker, reason, position_info).
        """
        exits: list[tuple[str, str, dict[str, Any]]] = []

        for ticker, pos in list(self.positions.items()):
            price = current_prices.get(ticker)
            if not price:
                continue

            pos["max_price"] = max(pos["max_price"], price)
            gain = (price - pos["entry_price"]) / pos["entry_price"]
            max_gain = (pos["max_price"] - pos["entry_price"]) / pos["entry_price"]

            params = pos["params"]
            tp_pct = params.get("tp_pct", 0.025)
            max_hold = int(params.get("max_hold", 12))
            min_hold = int(params.get("min_hold", 3))
            trail_trigger = params.get("trail_trigger", 0.015)
            trail_exit = params.get("trail_exit", 0.005)

            # TP
            exit_info = {**pos, "exit_price": price, "pnl_pct": gain * 100}
            if gain >= tp_pct:
                exits.append((ticker, "take_profit", exit_info))
                continue

            if pos["days_held"] < min_hold:
                continue

            # Time exit
            if pos["days_held"] >= max_hold:
                exits.append((ticker, "time_exit", exit_info))
                continue

            # Trail
            if max_gain >= trail_trigger and gain < trail_exit:
                exits.append((ticker, "trailing_stop", exit_info))

        return exits

    def increment_day(self) -> None:
        """Called at end of each trading day."""
        for pos in self.positions.values():
            pos["days_held"] += 1


# ── Main Scanner Loop ────────────────────────────────────────────────────────


async def main(
    dry_run: bool = False,
    ticker_filter: list[str] | None = None,
    max_positions: int = MAX_TOTAL_POSITIONS,
) -> None:
    """Run the specialist ensemble scanner."""
    load_dotenv()
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    # Load specialist configs
    all_configs = _load_specialist_configs()
    if ticker_filter:
        all_configs = [c for c in all_configs if c["tickers"][0] in ticker_filter]
    logger.info("specialists_loaded", count=len(all_configs))

    if not all_configs:
        logger.error("no_specialists", msg="No specialist configs found")
        return

    all_tickers = sorted(set(c["tickers"][0] for c in all_configs))
    config_by_ticker: dict[str, dict[str, Any]] = {
        c["tickers"][0]: c for c in all_configs
    }

    # Init Alpaca — Account 4 (Zeus)
    api_key = os.getenv("ZEUS_ALPACA_KEY_ID", os.getenv("ALPACA_API_KEY_ID", ""))
    secret_key = os.getenv("ZEUS_ALPACA_SECRET_KEY", os.getenv("ALPACA_API_SECRET_KEY", ""))
    if not api_key or not secret_key:
        logger.error("missing_alpaca_keys")
        return

    executor = AlpacaExecutor(api_key, secret_key, paper=True)
    polygon = PolygonIntradayProvider(os.getenv("POLYGON_API_KEY", ""))
    tracker = PositionTracker()

    logger.info(
        "zeus_scanner_started",
        tickers=len(all_tickers),
        max_positions=max_positions,
        dry_run=dry_run,
    )

    try:
        while True:
            now = datetime.now(ET_OFFSET)
            current_time = now.time()

            # Outside market hours — sleep
            if current_time < MARKET_OPEN or current_time > MARKET_CLOSE:
                if current_time > MARKET_CLOSE:
                    tracker.increment_day()
                    logger.info("market_closed", next_open="tomorrow 9:35 AM ET")
                await asyncio.sleep(60)
                continue

            # ── Exit monitoring (every 30s when positions open) ──
            if tracker.positions:
                try:
                    current_prices = await _get_current_prices(
                        polygon, list(tracker.positions.keys()),
                    )
                    exits = tracker.check_exits(current_prices)

                    for ticker, reason, pos_info in exits:
                        if not dry_run:
                            await executor.sell_option(
                                ticker, pos_info["qty"],
                                model_name=MODEL_NAME,
                                reason=reason,
                            )
                        tracker.remove(ticker)
                        _log_exit(ticker, reason, pos_info)
                        logger.info(
                            "zeus_exit",
                            ticker=ticker,
                            reason=reason,
                            pnl_pct=f"{pos_info['pnl_pct']:+.1f}%",
                            hold_days=pos_info["days_held"],
                            specialist=pos_info["specialist"],
                        )
                except Exception as e:
                    logger.error("exit_check_failed", error=str(e)[:200])

            # ── Entry scan (near market close, every 5 min) ──
            if (
                current_time >= ENTRY_SCAN_TIME
                and len(tracker.positions) < max_positions
            ):
                try:
                    signals = await _scan_all_specialists(
                        polygon, all_tickers, config_by_ticker,
                    )

                    # Rank by specialist WR (conviction)
                    signals.sort(key=lambda s: s["conviction"], reverse=True)

                    # Apply position limits
                    current_sectors: dict[str, int] = {}
                    for pos in tracker.positions.values():
                        sector = SECTOR_MAP.get(pos["ticker"], "other")
                        current_sectors[sector] = current_sectors.get(sector, 0) + 1

                    for sig in signals:
                        if len(tracker.positions) >= max_positions:
                            break

                        ticker = sig["ticker"]
                        config = config_by_ticker[ticker]
                        params = config["shares_params"]
                        cooldown = int(params.get("cooldown", 3))

                        if not tracker.can_enter(ticker, cooldown):
                            continue

                        sector = SECTOR_MAP.get(ticker, "other")
                        if current_sectors.get(sector, 0) >= MAX_PER_SECTOR:
                            continue

                        # Size position
                        if not dry_run:
                            account = await executor.get_account()
                            equity = float(account.get("equity", 10000))
                        else:
                            equity = 100_000.0

                        risk_pct = params.get("risk_pct", 0.10)
                        budget = equity * risk_pct
                        price = sig["close"]
                        qty = int(budget / price)
                        if qty < 1:
                            continue

                        # Execute
                        if not dry_run:
                            order = await executor.buy_option(
                                ticker, qty,
                                order_type="market",
                                model_name=MODEL_NAME,
                                conviction=sig["conviction"],
                            )
                            logger.info(
                                "zeus_entry",
                                ticker=ticker,
                                qty=qty,
                                price=price,
                                conviction=sig["conviction"],
                                specialist=config["name"],
                                order_status=order.status,
                            )
                        else:
                            logger.info(
                                "zeus_signal_DRY",
                                ticker=ticker,
                                qty=qty,
                                price=price,
                                conviction=sig["conviction"],
                                specialist=config["name"],
                                ibs=sig["ibs"],
                                rsi=sig["rsi"],
                            )

                        tracker.add(ticker, price, qty, params, config["name"])
                        current_sectors[sector] = current_sectors.get(sector, 0) + 1
                        _log_entry(ticker, sig, config, qty)

                except Exception as e:
                    logger.error("scan_failed", error=str(e)[:200])

            # Sleep
            sleep_time = EXIT_CHECK_INTERVAL if tracker.positions else SCAN_INTERVAL
            await asyncio.sleep(sleep_time)

    except KeyboardInterrupt:
        logger.info("scanner_stopped")
    finally:
        await executor.close()


# ── Data Helpers ─────────────────────────────────────────────────────────────


async def _scan_all_specialists(
    polygon: PolygonIntradayProvider,
    tickers: list[str],
    config_by_ticker: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Pull daily bars and check entry signals for all specialists."""
    signals: list[dict[str, Any]] = []

    # Pull last 60 daily bars for each ticker
    to_date = date.today()
    from_date = to_date - timedelta(days=90)  # ~60 trading days

    for ticker in tickers:
        try:
            bars = await polygon.get_intraday_bars(
                ticker, Timeframe.DAILY, from_date, to_date, limit=60,
            )
            if not bars or len(bars) < 55:
                continue

            daily = [
                {
                    "date": b.timestamp[:10] if hasattr(b, "timestamp") else str(b.date),
                    "open": b.open,
                    "high": b.high,
                    "low": b.low,
                    "close": b.close,
                    "volume": b.volume,
                }
                for b in bars
            ]

            config = config_by_ticker.get(ticker)
            if not config:
                continue

            params = config["shares_params"]
            signal = check_specialist_signal(daily, params)
            if signal:
                signal["ticker"] = ticker
                signal["conviction"] = config.get("optimized_win_rate", 0.7) * 10
                signal["specialist"] = config["name"]
                signals.append(signal)

        except Exception as e:
            logger.debug("ticker_scan_error", ticker=ticker, error=str(e)[:100])

    return signals


async def _get_current_prices(
    polygon: PolygonIntradayProvider,
    tickers: list[str],
) -> dict[str, float]:
    """Get current prices for position monitoring."""
    prices: dict[str, float] = {}
    for ticker in tickers:
        try:
            snapshot = await polygon.get_snapshot(ticker)
            if snapshot:
                prices[ticker] = float(snapshot.get("lastTrade", {}).get("p", 0))
        except Exception:
            pass
    return prices


# ── Logging ──────────────────────────────────────────────────────────────────


def _log_entry(
    ticker: str,
    signal: dict[str, Any],
    config: dict[str, Any],
    qty: int,
) -> None:
    """Log entry to daily file."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_file = LOG_DIR / f"entries_{date.today().isoformat()}.jsonl"
    entry = {
        "timestamp": datetime.now().isoformat(),
        "type": "entry",
        "ticker": ticker,
        "qty": qty,
        "signal": signal,
        "specialist": config["name"],
        "params": config["shares_params"],
    }
    with log_file.open("a") as f:
        f.write(json.dumps(entry, default=str) + "\n")


def _log_exit(
    ticker: str,
    reason: str,
    pos_info: dict[str, Any],
) -> None:
    """Log exit to daily file."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_file = LOG_DIR / f"exits_{date.today().isoformat()}.jsonl"
    entry = {
        "timestamp": datetime.now().isoformat(),
        "type": "exit",
        "ticker": ticker,
        "reason": reason,
        "entry_price": pos_info["entry_price"],
        "exit_price": pos_info.get("exit_price", 0),
        "pnl_pct": pos_info.get("pnl_pct", 0),
        "hold_days": pos_info["days_held"],
        "specialist": pos_info["specialist"],
    }
    with log_file.open("a") as f:
        f.write(json.dumps(entry, default=str) + "\n")


# ── CLI ──────────────────────────────────────────────────────────────────────


if __name__ == "__main__":
    asyncio.run(main())
