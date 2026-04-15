"""Lotto Candlestick Live Scanner — YOLO and Production modes.

Detects candlestick reversal patterns on 5-min bars, executes 0-2 DTE
options via Alpaca, sends email alerts on every entry/exit.

  LOTTO_MODE=yolo       → Account 5 (YOLO_ALPACA_KEY_ID)
  LOTTO_MODE=production → Account 6 (PROD_LOTTO_ALPACA_KEY_ID)
"""
from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import smtplib
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from typing import Any

logger = logging.getLogger("lotto.live")
LOG_DIR = Path("data/live_logs/lotto")
LOG_DIR.mkdir(parents=True, exist_ok=True)

SMTP_HOST = os.getenv("ALERT_SMTP_HOST", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("ALERT_SMTP_PORT", "587"))
SMTP_USER = os.getenv("ALERT_SMTP_USER", "")
SMTP_PASS = os.getenv("ALERT_SMTP_PASS", "")
EMAIL_TO = os.getenv("ALERT_EMAIL_TO", "")
EMAIL_FROM = os.getenv("ALERT_EMAIL_FROM", SMTP_USER)

TICKERS = [
    "SPY", "QQQ", "IWM", "AAPL", "NVDA", "META", "TSLA", "MSFT", "AMD", "AMZN",
    "GOOGL", "PLTR", "MSTR", "COIN", "AVGO", "NFLX", "ARM", "SMCI", "CRM", "COST",
    "HOOD", "RDDT", "SOFI",
]


def detect_candle(bars: list[dict[str, Any]], idx: int) -> list[tuple[str, str]]:  # noqa: N802
    if idx < 4:
        return []
    b, p = bars[idx], bars[idx - 1]
    sigs: list[tuple[str, str]] = []
    body = abs(b["c"] - b["o"])
    lo_w = min(b["o"], b["c"]) - b["l"]
    hi_w = b["h"] - max(b["o"], b["c"])
    rng = b["h"] - b["l"]

    if body > 0 and lo_w >= 2 * body and hi_w < body * 0.5:
        sigs.append(("hammer", "bull"))
    if body > 0 and hi_w >= 2 * body and lo_w < body * 0.5:
        sigs.append(("inv_hammer", "bear"))
    if (
        b["c"] > b["o"]
        and p["c"] < p["o"]
        and b["o"] <= p["c"]
        and b["c"] >= p["o"]
    ):
        sigs.append(("bull_engulf", "bull"))
    if (
        b["c"] < b["o"]
        and p["c"] > p["o"]
        and b["o"] >= p["c"]
        and b["c"] <= p["o"]
    ):
        sigs.append(("bear_engulf", "bear"))
    if (
        b["c"] > b["o"]
        and bars[idx - 1]["c"] < bars[idx - 1]["o"]
        and bars[idx - 2]["c"] < bars[idx - 2]["o"]
        and bars[idx - 3]["c"] < bars[idx - 3]["o"]
    ):
        sigs.append(("3bar_bull", "bull"))
    if (
        b["c"] < b["o"]
        and bars[idx - 1]["c"] > bars[idx - 1]["o"]
        and bars[idx - 2]["c"] > bars[idx - 2]["o"]
        and bars[idx - 3]["c"] > bars[idx - 3]["o"]
    ):
        sigs.append(("3bar_bear", "bear"))
    if rng > 0 and body / rng < 0.25:
        if lo_w > hi_w * 2:
            sigs.append(("pin", "bull"))
        elif hi_w > lo_w * 2:
            sigs.append(("pin", "bear"))
    if idx >= 4:
        pr = sum(abs(bars[i]["h"] - bars[i]["l"]) for i in range(idx - 3, idx)) / 3
        if pr > 0 and rng > pr * 2 and body > pr * 1.5:
            sigs.append(("expansion", "bull" if b["c"] > b["o"] else "bear"))
    return sigs


def _send_email(subject: str, html: str) -> None:
    if not SMTP_USER or not SMTP_PASS:
        return
    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"] = f"FlowEdge Lotto <{EMAIL_FROM}>"
        msg["To"] = EMAIL_TO
        msg.attach(MIMEText(html, "html"))
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=10) as s:
            s.starttls()
            s.login(SMTP_USER, SMTP_PASS)
            s.sendmail(EMAIL_FROM, EMAIL_TO, msg.as_string())
    except Exception as e:
        logger.warning("Email failed: %s", e)


@dataclass
class Pos:
    ticker: str
    is_call: bool
    option_symbol: str
    entry_price: float
    peak_price: float
    contracts: int
    entry_time: datetime
    bars_held: int = field(default=0)


class AlpacaClient:
    def __init__(self, key: str, secret: str) -> None:
        self.base = "https://paper-api.alpaca.markets"
        self.headers = {
            "APCA-API-KEY-ID": key,
            "APCA-API-SECRET-KEY": secret,
            "Content-Type": "application/json",
        }

    async def _req(
        self, method: str, path: str, body: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        import urllib.request

        url = f"{self.base}{path}"
        data = json.dumps(body).encode() if body else None
        req = urllib.request.Request(url, data=data, headers=self.headers, method=method)
        loop = asyncio.get_event_loop()
        resp = await loop.run_in_executor(
            None, lambda: urllib.request.urlopen(req, timeout=15)
        )
        return json.loads(resp.read())  # type: ignore[no-any-return]

    async def get_account(self) -> dict[str, Any]:
        return await self._req("GET", "/v2/account")

    async def get_positions(self) -> list[dict[str, Any]]:
        return await self._req("GET", "/v2/positions")  # type: ignore[return-value]

    async def submit_order(
        self, symbol: str, qty: int, side: str, cid: str
    ) -> dict[str, Any]:
        return await self._req(
            "POST",
            "/v2/orders",
            {
                "symbol": symbol,
                "qty": str(qty),
                "side": side,
                "type": "market",
                "time_in_force": "day",
                "client_order_id": cid,
            },
        )

    async def close_position(self, symbol: str) -> dict[str, Any]:
        return await self._req("DELETE", f"/v2/positions/{symbol}")


class LottoScanner:
    def __init__(self, mode: str = "yolo") -> None:
        self.mode = mode
        if mode == "yolo":
            self.model_tag = "lotto_yolo"
            self.risk_pct = 0.20
            self.max_positions = 999
            self.daily_loss_limit = 1.0
            key = os.getenv("YOLO_ALPACA_KEY_ID", "")
            secret = os.getenv("YOLO_ALPACA_SECRET_KEY", "")
        else:
            self.model_tag = "lotto_prod"
            self.risk_pct = 0.10
            self.max_positions = 10
            self.daily_loss_limit = 0.15
            key = os.getenv("PROD_LOTTO_ALPACA_KEY_ID", "")
            secret = os.getenv("PROD_LOTTO_ALPACA_SECRET_KEY", "")
        if not key or not secret:
            raise ValueError(f"Alpaca keys not set for {mode}")
        self.alpaca = AlpacaClient(key, secret)
        self.polygon_key = os.getenv("POLYGON_API_KEY", "")
        self.positions: dict[str, Pos] = {}
        self.daily_pnl = 0.0
        self.trade_count = 0
        self.today = ""
        self.tp_pct = 0.30
        self.sl_pct = -0.30
        self.max_hold_bars = 4
        self.trail_pct = 0.35
        self.log_path = LOG_DIR / f"lotto_{mode}_{datetime.now(UTC).strftime('%Y-%m-%d')}.jsonl"

    def _log(self, event: str, data: dict[str, Any]) -> None:
        with open(self.log_path, "a") as f:
            f.write(
                json.dumps(
                    {"ts": datetime.now(UTC).isoformat(), "event": event, **data},
                    default=str,
                )
                + "\n"
            )

    async def _fetch_bars(self, ticker: str) -> list[dict[str, Any]]:
        import urllib.request

        today = datetime.now(UTC).strftime("%Y-%m-%d")
        url = (
            f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/minute"
            f"/{today}/{today}?adjusted=true&sort=asc&limit=50000"
            f"&apiKey={self.polygon_key}"
        )
        loop = asyncio.get_event_loop()
        try:
            req = urllib.request.Request(url)
            resp = await loop.run_in_executor(
                None, lambda: urllib.request.urlopen(req, timeout=15)
            )
            data = json.loads(resp.read())
            return [
                {
                    "ts": b["t"] * 1_000_000,
                    "o": b["o"],
                    "h": b["h"],
                    "l": b["l"],
                    "c": b["c"],
                    "v": b["v"],
                }
                for b in data.get("results", [])
            ]
        except Exception:
            return []

    def _agg_5m(self, bars: list[dict[str, Any]]) -> list[dict[str, Any]]:
        window = 5 * 60 * 1_000_000
        buckets: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for b in bars:
            buckets[b["ts"] // window].append(b)
        return [
            {
                "ts": c[0]["ts"],
                "o": c[0]["o"],
                "h": max(x["h"] for x in c),
                "l": min(x["l"] for x in c),
                "c": c[-1]["c"],
                "v": sum(x["v"] for x in c),
            }
            for _, c in sorted(buckets.items())
        ]

    async def _find_option(
        self, ticker: str, price: float, opt_type: str
    ) -> dict[str, Any] | None:
        import urllib.request

        exp_date = datetime.now(UTC).strftime("%Y-%m-%d")
        ct = "call" if opt_type == "C" else "put"
        url = (
            f"https://api.polygon.io/v3/snapshot/options/{ticker}"
            f"?strike_price.gte={price * 0.97:.2f}"
            f"&strike_price.lte={price * 1.03:.2f}"
            f"&expiration_date.gte={exp_date}"
            f"&contract_type={ct}&limit=10&apiKey={self.polygon_key}"
        )
        loop = asyncio.get_event_loop()
        try:
            req = urllib.request.Request(url)
            resp = await loop.run_in_executor(
                None, lambda: urllib.request.urlopen(req, timeout=15)
            )
            data = json.loads(resp.read())
            results = data.get("results", [])
            if not results:
                return None
            best = max(results, key=lambda r: r.get("day", {}).get("volume", 0))
            d = best.get("details", {})
            q = best.get("last_quote", {})
            return {
                "symbol": d.get("ticker", ""),
                "strike": d.get("strike_price", 0),
                "expiry": d.get("expiration_date", ""),
                "ask": q.get("ask", 0),
                "bid": q.get("bid", 0),
                "last": q.get(
                    "last_price", best.get("last_trade", {}).get("price", 0)
                ),
                "volume": best.get("day", {}).get("volume", 0),
            }
        except Exception:
            return None

    async def check_exits(self) -> None:
        if not self.positions:
            return
        try:
            alp_pos = await self.alpaca.get_positions()
        except Exception:
            return
        for ticker in list(self.positions.keys()):
            pos = self.positions[ticker]
            alp = next(
                (p for p in alp_pos if p["symbol"] == pos.option_symbol), None
            )
            if not alp:
                del self.positions[ticker]
                continue
            current = float(alp.get("current_price", 0))
            entry = pos.entry_price
            pos.bars_held += 1
            if current > pos.peak_price:
                pos.peak_price = current
            reason = None
            if entry > 0:
                pnl = (current - entry) / entry
                if pnl >= self.tp_pct:
                    reason = "tp"
                elif pnl <= self.sl_pct:
                    reason = "sl"
                elif pos.bars_held >= self.max_hold_bars:
                    reason = "time"
                elif (
                    pos.peak_price > entry
                    and (pos.peak_price - current) / pos.peak_price >= self.trail_pct
                ):
                    reason = "trail"
            if not reason:
                continue
            try:
                await self.alpaca.close_position(pos.option_symbol)
                pnl_pct = (current - entry) / entry * 100
                pnl_d = (current - entry) * pos.contracts * 100
                self.daily_pnl += pnl_d
                self.trade_count += 1
                logger.info(
                    "EXIT [%s] %s | %+.1f%% ($%+.2f)", reason, ticker, pnl_pct, pnl_d
                )
                self._log(
                    "exit",
                    {"ticker": ticker, "reason": reason, "pnl_pct": pnl_pct, "pnl_d": pnl_d},
                )
                color = "#10b981" if pnl_d > 0 else "#ef4444"
                label = "WIN" if pnl_d > 0 else "LOSS"
                _send_email(
                    f"[Lotto {self.mode.upper()}] {label} {ticker} {pnl_pct:+.1f}%",
                    f'<div style="font-family:sans-serif;max-width:400px;'
                    f'background:#0f172a;color:#e2e8f0;border-radius:12px;'
                    f'overflow:hidden"><div style="background:{color};'
                    f'padding:12px 16px"><b>Lotto {self.mode.upper()} Exit</b>'
                    f'</div><div style="padding:16px">'
                    f'<b style="font-size:24px">{ticker}</b><br>'
                    f'P&L: <b style="color:{color}">'
                    f"${pnl_d:+,.2f} ({pnl_pct:+.1f}%)</b><br>"
                    f"Reason: {reason} | Held: {pos.bars_held * 5}min"
                    f"</div></div>",
                )
                del self.positions[ticker]
            except Exception as e:
                logger.error("Exit failed %s: %s", ticker, e)

    async def scan_and_trade(self) -> None:
        today = datetime.now(UTC).strftime("%Y-%m-%d")
        if today != self.today:
            self.daily_pnl = 0.0
            self.today = today
        try:
            acct = await self.alpaca.get_account()
            equity = float(acct.get("equity", 25000))
        except Exception:
            equity = 25000
        if self.daily_pnl < 0 and abs(self.daily_pnl) / equity >= self.daily_loss_limit:
            return
        for ticker in TICKERS:
            if ticker in self.positions or len(self.positions) >= self.max_positions:
                continue
            bars_1m = await self._fetch_bars(ticker)
            if len(bars_1m) < 30:
                continue
            bars_5m = self._agg_5m(bars_1m)
            if len(bars_5m) < 10:
                continue
            sigs = detect_candle(bars_5m, len(bars_5m) - 1)
            if not sigs:
                continue
            bull = sum(1 for _, d in sigs if d == "bull")
            bear = sum(1 for _, d in sigs if d == "bear")
            is_call = bull >= bear
            opt_type = "C" if is_call else "P"
            price = bars_5m[-1]["c"]
            option = await self._find_option(ticker, price, opt_type)
            if not option or not option["symbol"]:
                continue
            ask = option["ask"] or option["last"]
            if ask < 0.10:
                continue
            contracts = max(1, int(equity * self.risk_pct / (ask * 100)))
            try:
                cid = f"{self.model_tag}_{ticker}_{datetime.now(UTC).strftime('%H%M%S')}"[:48]
                await self.alpaca.submit_order(option["symbol"], contracts, "buy", cid)
                self.positions[ticker] = Pos(
                    ticker=ticker,
                    is_call=is_call,
                    option_symbol=option["symbol"],
                    entry_price=ask,
                    peak_price=ask,
                    contracts=contracts,
                    entry_time=datetime.now(UTC),
                )
                patterns = [s[0] for s in sigs]
                arrow = "CALL" if is_call else "PUT"
                logger.info(
                    "ENTRY %s %s %s | %s | %d x $%.2f",
                    ticker, arrow, option["symbol"], patterns, contracts, ask,
                )
                self._log(
                    "entry",
                    {
                        "ticker": ticker,
                        "direction": arrow,
                        "option": option["symbol"],
                        "contracts": contracts,
                        "price": ask,
                        "patterns": patterns,
                    },
                )
                cost_str = f"${contracts * ask * 100:,.0f}"
                _send_email(
                    f"[Lotto {self.mode.upper()}] {arrow} {ticker}",
                    f'<div style="font-family:sans-serif;max-width:400px;'
                    f'background:#0f172a;color:#e2e8f0;border-radius:12px;'
                    f'overflow:hidden"><div style="background:#10b981;'
                    f'padding:12px 16px"><b>Lotto {self.mode.upper()} Entry'
                    f"</b></div>"
                    f'<div style="padding:16px">'
                    f'<b style="font-size:24px">{ticker}</b> {arrow}<br>'
                    f'Option: {option["symbol"]}<br>'
                    f"{contracts} x ${ask:.2f} = {cost_str}<br>"
                    f'Patterns: {", ".join(patterns)}</div></div>',
                )
            except Exception as e:
                logger.error("Order failed %s: %s", ticker, e)
            await asyncio.sleep(0.3)

    async def run(self) -> None:
        logger.info(
            "Lotto %s Scanner starting | %d tickers | risk=%.0f%%",
            self.mode.upper(), len(TICKERS), self.risk_pct * 100,
        )
        while True:
            now = datetime.now(UTC)
            et_hour = now.hour - 4
            mins = (et_hour - 9) * 60 + (now.minute - 30)
            if mins < 5 or mins > 390:
                if mins > 385 and self.positions:
                    for t in list(self.positions.keys()):
                        with contextlib.suppress(Exception):
                            await self.alpaca.close_position(
                                self.positions[t].option_symbol
                            )
                    self.positions.clear()
                await asyncio.sleep(60)
                continue
            await self.check_exits()
            await self.scan_and_trade()
            await asyncio.sleep(15 if self.positions else 60)


async def run_yolo() -> None:
    await LottoScanner("yolo").run()


async def run_production() -> None:
    await LottoScanner("production").run()
