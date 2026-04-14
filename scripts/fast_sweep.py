#!/usr/bin/env python3
"""Fast parameter sweep — loads data ONCE, then sweeps filter/exit params.

The expensive part is loading ~8 tickers × 1000 days of bars from disk.
This script loads once, then runs the filter/exit logic in a tight loop.
"""

from __future__ import annotations

import json
import sys
import time
from collections import defaultdict
from datetime import date
from itertools import product
from math import sqrt
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from flowedge.scanner.backtest.options_matcher import OptionsMatcher  # noqa: E402

CACHE_DIR = Path("data/flat_files_s3")
RISK_FREE_RATE = 0.05

TICKERS = ["IWM", "COST", "V", "INTC", "PLTR", "CRM", "WMT", "NVDA"]


def _is_dst(date_str: str) -> bool:
    try:
        d = date.fromisoformat(date_str)
    except (ValueError, TypeError):
        return True
    if d.month == 3:
        first = date(d.year, 3, 1)
        days_to_sunday = (6 - first.weekday()) % 7
        second_sunday = first.day + days_to_sunday + 7
        return d.day >= second_sunday
    if d.month == 11:
        first = date(d.year, 11, 1)
        days_to_sunday = (6 - first.weekday()) % 7
        first_sunday = first.day + days_to_sunday
        return d.day < first_sunday
    return 4 <= d.month <= 10


_RTH_EDT_START = 13 * 3600 + 25 * 60
_RTH_EDT_END = 20 * 3600 + 5 * 60
_RTH_EST_START = 14 * 3600 + 25 * 60
_RTH_EST_END = 21 * 3600 + 5 * 60


def _filter_rth(bars: list[dict], date_str: str) -> list[dict]:
    dst = _is_dst(date_str)
    rth_start = _RTH_EDT_START if dst else _RTH_EST_START
    rth_end = _RTH_EDT_END if dst else _RTH_EST_END
    rth = []
    for b in bars:
        ts_ns = int(b.get("ts", b.get("timestamp", 0)))
        if ts_ns == 0:
            continue
        ts_sec = ts_ns // 1_000_000_000
        secs_into_day = ts_sec % 86400
        if rth_start <= secs_into_day <= rth_end:
            rth.append(b)
    return rth


def _gf(bar: dict, long_key: str, short_key: str) -> float:
    return float(bar.get(long_key, bar.get(short_key, 0)))


# ── Pre-compute all data that doesn't depend on parameters ────────

def load_all_data(tickers: list[str]) -> dict:
    """Load bars and build 5-min chunks once."""
    print("Loading data...", end=" ", flush=True)
    t0 = time.time()

    all_bars: dict[str, dict[str, list[dict]]] = {}
    for ticker in tickers:
        min_dir = CACHE_DIR / ticker / "1min"
        if not min_dir.exists():
            continue
        bars: list[dict] = []
        for f in sorted(min_dir.glob("*.json")):
            bars.extend(json.loads(f.read_text()))
        # Basic validation
        clean = []
        for b in bars:
            h = float(b.get("high", b.get("h", 0)))
            c = float(b.get("close", b.get("c", 0)))
            if h > 0 and c > 0:
                clean.append(b)
        if len(clean) < 5000:
            continue
        by_date: dict[str, list[dict]] = defaultdict(list)
        for b in clean:
            d = str(b.get("date", b.get("d", "")))
            if d:
                by_date[d].append(b)
        all_bars[ticker] = dict(by_date)

    # Get sorted dates
    all_dates: set[str] = set()
    for ticker_days in all_bars.values():
        all_dates.update(ticker_days.keys())
    sorted_dates = sorted(all_dates)

    # Pre-build daily closes and 5-min chunks per ticker per day
    daily_closes: dict[str, list[float]] = {t: [] for t in all_bars}
    chunks_by_day: dict[str, dict[str, list[dict]]] = {
        t: {} for t in all_bars
    }
    vwaps_by_day: dict[str, dict[str, list[float]]] = {
        t: {} for t in all_bars
    }
    day_opens_by_day: dict[str, dict[str, float]] = {
        t: {} for t in all_bars
    }

    for d in sorted_dates:
        for ticker in all_bars:
            raw_day = all_bars[ticker].get(d, [])
            if raw_day:
                daily_closes[ticker].append(
                    _gf(raw_day[-1], "close", "c")
                )

            day_bars = _filter_rth(raw_day, date_str=d)
            if len(day_bars) < 50:
                continue

            dc = daily_closes[ticker]
            if len(dc) < 20:
                continue

            sma10 = sum(dc[-10:]) / 10
            sma20 = sum(dc[-20:]) / 20
            if sma10 <= sma20:
                continue

            # Build 5-min chunks
            _5min_ns = 5 * 60 * 1_000_000_000
            window_buckets: dict[int, list[dict]] = defaultdict(list)
            for b in day_bars:
                ts_ns = int(b.get("ts", b.get("timestamp", 0)))
                if ts_ns == 0:
                    continue
                bucket = ts_ns // _5min_ns
                window_buckets[bucket].append(b)

            chunks = []
            for bucket in sorted(window_buckets):
                chunk = window_buckets[bucket]
                o = _gf(chunk[0], "open", "o")
                h = max(_gf(b, "high", "h") for b in chunk)
                lows = [_gf(b, "low", "l") for b in chunk if _gf(b, "low", "l") > 0]
                lo = min(lows) if lows else 0.0
                c = _gf(chunk[-1], "close", "c")
                v = sum(_gf(b, "volume", "v") for b in chunk)
                ts = str(chunk[0].get("ts", chunk[0].get("timestamp", "")))
                chunks.append({"o": o, "h": h, "l": lo, "c": c, "v": v, "ts": ts})

            if len(chunks) < 30:
                continue

            chunks_by_day[ticker][d] = chunks
            day_opens_by_day[ticker][d] = chunks[0]["o"]

            # VWAP
            cum_pv = 0.0
            cum_v = 0.0
            vwaps = []
            for ch in chunks:
                tp = (ch["h"] + ch["l"] + ch["c"]) / 3
                cum_pv += tp * ch["v"]
                cum_v += ch["v"]
                vwaps.append(cum_pv / cum_v if cum_v > 0 else ch["c"])
            vwaps_by_day[ticker][d] = vwaps

    elapsed = time.time() - t0
    total_days = sum(len(v) for v in chunks_by_day.values())
    print(f"done ({elapsed:.1f}s, {total_days} ticker-days)")

    return {
        "sorted_dates": sorted_dates,
        "chunks_by_day": chunks_by_day,
        "vwaps_by_day": vwaps_by_day,
        "day_opens_by_day": day_opens_by_day,
    }


# ── Fast backtest with given parameters ───────────────────────────

def run_fast_backtest(
    data: dict,
    matcher: OptionsMatcher,
    *,
    ibs: float = 0.10,
    rsi3: float = 20.0,
    vol_spike: float = 2.0,
    intraday_drop: float = -0.003,
    max_hold: int = 12,
    trail_pct: float = 0.05,
    tp_und: float = 0.0015,
    min_premium: float = 0.30,
    max_positions: int = 2,
    risk_pct: float = 0.05,
    commission: float = 0.50,
    dte: int = 5,
    starting_capital: float = 25_000.0,
) -> dict:
    """Run the full backtest logic using pre-loaded data."""
    sorted_dates = data["sorted_dates"]
    chunks_by_day = data["chunks_by_day"]
    vwaps_by_day = data["vwaps_by_day"]
    day_opens_by_day = data["day_opens_by_day"]

    cash = starting_capital
    trades: list[dict] = []
    daily_values: list[tuple[str, float]] = []
    signals_total = 0

    for d in sorted_dates:
        intraday_positions: list[dict] = []

        for ticker in chunks_by_day:
            chunks = chunks_by_day[ticker].get(d)
            if chunks is None:
                continue
            vwaps = vwaps_by_day[ticker].get(d)
            if vwaps is None:
                continue
            day_open = day_opens_by_day[ticker].get(d, 0)
            if day_open <= 0:
                continue

            for i in range(6, min(24, len(chunks) - max_hold)):
                if len(intraday_positions) >= max_positions:
                    break
                if any(p["ticker"] == ticker for p in intraday_positions):
                    break

                ch = chunks[i]
                rng = ch["h"] - ch["l"]
                if rng <= 0 or ch["c"] <= 0:
                    continue

                # 7-condition filter
                ibs_val = (ch["c"] - ch["l"]) / rng
                if ibs_val >= ibs:
                    continue

                if i < 4:
                    continue
                c5m = [chunks[j]["c"] for j in range(i - 3, i + 1)]
                g = [max(0, c5m[k] - c5m[k - 1]) for k in range(1, 4)]
                ls = [max(0, c5m[k - 1] - c5m[k]) for k in range(1, 4)]
                ag = sum(g) / 3
                al = sum(ls) / 3
                rsi3_val = 100.0 - 100.0 / (1 + ag / al) if al > 0 else 100.0
                if rsi3_val >= rsi3:
                    continue

                if ch["c"] >= vwaps[i]:
                    continue

                start_idx = max(0, i - 10)
                avg_vol = sum(chunks[j]["v"] for j in range(start_idx, i)) / max(1, i - start_idx)
                vr = ch["v"] / avg_vol if avg_vol > 0 else 1
                if vr < vol_spike:
                    continue

                drop = (ch["c"] - day_open) / day_open
                if drop > intraday_drop:
                    continue

                if i > 0 and chunks[i - 1]["c"] >= chunks[max(0, i - 2)]["c"]:
                    continue

                if i >= 10:
                    sma5 = sum(chunks[j]["c"] for j in range(i - 4, i + 1)) / 5
                    sma10_id = (
                        sum(chunks[j]["c"] for j in range(max(0, i - 9), i + 1))
                        / min(10, i + 1)
                    )
                    if sma5 >= sma10_id:
                        continue

                # ALL 7 passed
                signals_total += 1
                entry_price = ch["c"]
                signal_ts = int(ch["ts"]) if ch["ts"] else 0
                if signal_ts == 0:
                    continue

                contract = matcher.find_best_contract(
                    underlying=ticker, date_str=d,
                    underlying_price=entry_price,
                    signal_ts_ns=signal_ts, max_dte=dte,
                )
                if contract is None:
                    continue

                opt_5min = OptionsMatcher.aggregate_to_5min(contract.bars)
                if not opt_5min:
                    continue

                next_ts = signal_ts + 5 * 60 * 1_000_000_000
                entry_bar = OptionsMatcher.get_bar_at_time(
                    opt_5min, next_ts, tolerance_ns=180_000_000_000,
                )
                if entry_bar is None:
                    continue
                fill = float(entry_bar.get("o", 0))

                if fill < min_premium:
                    continue

                budget = cash * risk_pct
                contracts = max(1, int(budget / (fill * 100)))
                cost = contracts * fill * 100
                entry_comm = contracts * commission

                if cost > cash * 0.9 or cost < 10:
                    continue

                # Hold simulation
                hold_start_ts = int(entry_bar.get("ts", signal_ts))
                hold_bars = OptionsMatcher.get_bars_after(opt_5min, hold_start_ts, max_hold + 1)

                max_prem = fill
                exit_fill = fill
                exit_reason = "time_exit"
                _exit_und = entry_price
                exit_bar_idx = 0

                for j, ob in enumerate(hold_bars[1:], start=1):
                    bc = float(ob.get("c", 0))
                    bh = float(ob.get("h", 0))
                    bl = float(ob.get("l", 0))
                    if bc <= 0:
                        continue
                    if bh > 0:
                        max_prem = max(max_prem, bh)

                    opt_ts = int(ob.get("ts", 0))
                    ub = None
                    if opt_ts > 0:
                        for ci in range(i, min(i + j + 2, len(chunks))):
                            ct = int(chunks[ci].get("ts", 0))
                            if ct >= opt_ts:
                                ub = chunks[ci]
                                break
                        if ub is None and i + j < len(chunks):
                            ub = chunks[i + j]

                    if ub is not None:
                        ug = (ub["c"] - entry_price) / entry_price
                        if ug >= tp_und:
                            exit_bar_idx = j
                            exit_reason = "take_profit"
                            _exit_und = ub["c"]
                            exit_fill = bc
                            break

                    if max_prem > fill * 1.05:
                        trail = max_prem * (1 - trail_pct)
                        if bl > 0 and bl <= trail:
                            exit_bar_idx = j
                            exit_reason = "trailing_stop"
                            if ub:
                                _exit_und = ub["c"]
                            exit_fill = bc
                            break

                    if j >= max_hold:
                        exit_bar_idx = j
                        exit_reason = "time_exit"
                        if ub:
                            _exit_und = ub["c"]
                        exit_fill = bc
                        break
                else:
                    if hold_bars:
                        exit_fill = float(hold_bars[-1].get("c", fill))

                exit_fill = max(0.01, exit_fill)
                exit_val = exit_fill * contracts * 100
                exit_comm = contracts * commission
                total_comm = entry_comm + exit_comm
                pnl = exit_val - cost - total_comm
                pnl_pct = (pnl / cost * 100) if cost > 0 else 0.0
                cash += pnl

                trades.append({
                    "ticker": ticker,
                    "date": d,
                    "pnl_pct": pnl_pct,
                    "pnl_dollars": pnl,
                    "exit_reason": exit_reason,
                    "hold_bars": exit_bar_idx,
                    "entry_price": fill,
                    "cost": cost + total_comm,
                    "exit_val": exit_val,
                    "year": d[:4],
                })

        daily_values.append((d, cash))

    # Compute metrics
    total = len(trades)
    wins = sum(1 for t in trades if t["pnl_pct"] >= 3)
    wr = wins / total if total > 0 else 0
    gp = sum(t["pnl_pct"] for t in trades if t["pnl_pct"] > 0)
    gl = abs(sum(t["pnl_pct"] for t in trades if t["pnl_pct"] < 0))
    pf = gp / gl if gl > 0 else 0

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
            for i in range(1, len(daily_values))
            if daily_values[i - 1][1] > 0
        ]
        if rets:
            m = sum(rets) / len(rets)
            var = sum((r - m) ** 2 for r in rets) / len(rets)
            s = sqrt(var) if var > 0 else 0.001
            sharpe = round((m * 252 - RISK_FREE_RATE) / (s * sqrt(252)), 3)

    # Walk-forward
    cutoff = "2023-12-31"
    train = [t for t in trades if t["date"] <= cutoff]
    val = [t for t in trades if t["date"] > cutoff]
    train_wr = sum(1 for t in train if t["pnl_pct"] >= 3) / len(train) if train else 0
    val_wr = sum(1 for t in val if t["pnl_pct"] >= 3) / len(val) if val else 0

    # By year
    by_year: dict[str, dict] = {}
    for yr in sorted(set(t["year"] for t in trades)):
        yt = [t for t in trades if t["year"] == yr]
        yw = sum(1 for t in yt if t["pnl_pct"] >= 3)
        by_year[yr] = {
            "trades": len(yt),
            "wr": round(yw / len(yt), 3) if yt else 0,
            "pnl": round(sum(t["pnl_dollars"] for t in yt)),
        }

    return {
        "trades": total,
        "wins": wins,
        "wr": round(wr, 3),
        "pnl": round(ending - starting_capital),
        "ret_pct": round(ret, 2),
        "pf": round(pf, 2),
        "max_dd": round(max_dd, 2),
        "sharpe": sharpe,
        "signals": signals_total,
        "train_wr": round(train_wr, 3),
        "val_wr": round(val_wr, 3),
        "wf_drift": round(abs(train_wr - val_wr), 3),
        "by_year": by_year,
    }


def main() -> None:
    print("=" * 70)
    print("FAST PARAMETER SWEEP — 8-Ticker Universe")
    print("=" * 70)

    # Load once
    data = load_all_data(TICKERS)
    matcher = OptionsMatcher()

    # Parameter grid — focused around winning region
    grid = list(product(
        [0.05, 0.07, 0.08, 0.10, 0.12],     # ibs
        [12.0, 15.0, 18.0, 20.0, 25.0],      # rsi3
        [1.5, 2.0, 2.5, 3.0],                # vol_spike
        [-0.002, -0.003, -0.004, -0.005],     # drop
        [6, 8, 10, 12],                       # max_hold
        [0.02, 0.03, 0.04, 0.05],            # trail
        [0.001, 0.0012, 0.0015, 0.002],      # tp_und
    ))

    print(f"Grid: {len(grid)} combinations")
    print()

    results = []
    t_start = time.time()

    for idx, (ibs_v, rsi_v, vs_v, drop_v, mh_v, tr_v, tp_v) in enumerate(grid):
        r = run_fast_backtest(
            data, matcher,
            ibs=ibs_v, rsi3=rsi_v, vol_spike=vs_v,
            intraday_drop=drop_v, max_hold=mh_v, trail_pct=tr_v,
            tp_und=tp_v,
        )

        if r["trades"] >= 5:
            r.update({
                "ibs": ibs_v, "rsi3": rsi_v, "vs": vs_v, "drop": drop_v,
                "mh": mh_v, "tr": tr_v, "tp": tp_v,
            })
            results.append(r)

        if (idx + 1) % 1000 == 0:
            elapsed = time.time() - t_start
            rate = (idx + 1) / elapsed
            remaining = (len(grid) - idx - 1) / rate
            best = max((r["wr"] for r in results), default=0) if results else 0
            print(
                f"  {idx+1:>5}/{len(grid)}  "
                f"{rate:.0f}/s  ETA {remaining:.0f}s  "
                f"best WR={best:.0%}  "
                f"valid={len(results)}"
            )

    total_time = time.time() - t_start
    print(f"\nDone: {len(grid)} combos in {total_time:.0f}s ({len(grid)/total_time:.0f}/s)")
    print(f"Valid results (5+ trades): {len(results)}")

    # Sort
    results.sort(key=lambda x: (x["wr"], x["trades"], x["pnl"]), reverse=True)

    # Top 25 by WR
    print(f"\n{'='*90}")
    print("TOP 25 BY WIN RATE (min 5 trades)")
    print(f"{'='*90}")
    for i, r in enumerate(results[:25]):
        tag = "***" if r["wr"] >= 0.75 else ("**" if r["wr"] >= 0.70 else "")
        print(
            f"{i+1:>3}. IBS={r['ibs']:.2f} RSI={r['rsi3']:>4.0f} VS={r['vs']:.1f} "
            f"D={r['drop']:.3f} MH={r['mh']:>2} TR={r['tr']:.2f} TP={r['tp']:.4f}  "
            f"{r['trades']:>2}t  WR {r['wr']:.0%}{tag}  "
            f"${r['pnl']:>+7,}  PF {r['pf']:.1f}  "
            f"Sh {r['sharpe']:>+.2f}  DD {r['max_dd']:.1f}%  "
            f"WF {r['train_wr']:.0%}/{r['val_wr']:.0%}"
        )

    # Best with 15+ trades
    vol = [r for r in results if r["trades"] >= 15 and r["pnl"] > 0]
    vol.sort(key=lambda x: (x["wr"], x["pnl"]), reverse=True)
    if vol:
        print(f"\n{'='*90}")
        print(f"BEST WITH 15+ TRADES AND PROFITABLE ({len(vol)} found)")
        print(f"{'='*90}")
        for i, r in enumerate(vol[:15]):
            print(
                f"{i+1:>3}. IBS={r['ibs']:.2f} RSI={r['rsi3']:>4.0f} VS={r['vs']:.1f} "
                f"D={r['drop']:.3f} MH={r['mh']:>2} TR={r['tr']:.2f} TP={r['tp']:.4f}  "
                f"{r['trades']:>2}t  WR {r['wr']:.0%}  "
                f"${r['pnl']:>+7,}  PF {r['pf']:.1f}  "
                f"Sh {r['sharpe']:>+.2f}  "
                f"WF {r['train_wr']:.0%}→{r['val_wr']:.0%} (Δ{r['wf_drift']:.0%})"
            )

    # Best balanced (WR * sqrt(trades) * log(pnl))
    from math import log
    balanced = [r for r in results if r["trades"] >= 10 and r["pnl"] > 0]
    for r in balanced:
        r["score"] = r["wr"] * sqrt(r["trades"]) * (1 + log(max(1, r["pnl"])) / 10)
    balanced.sort(key=lambda x: x["score"], reverse=True)
    if balanced:
        print(f"\n{'='*90}")
        print(f"BEST BALANCED (WR × √trades × log(PnL)) — {len(balanced)} found")
        print(f"{'='*90}")
        for i, r in enumerate(balanced[:10]):
            print(
                f"{i+1:>3}. IBS={r['ibs']:.2f} RSI={r['rsi3']:>4.0f} VS={r['vs']:.1f} "
                f"D={r['drop']:.3f} MH={r['mh']:>2} TR={r['tr']:.2f} TP={r['tp']:.4f}  "
                f"{r['trades']:>2}t  WR {r['wr']:.0%}  "
                f"${r['pnl']:>+7,}  PF {r['pf']:.1f}  "
                f"score={r['score']:.2f}  "
                f"WF {r['train_wr']:.0%}→{r['val_wr']:.0%}"
            )

    # Save
    out = Path("data/backtest_results/fast_sweep_results.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results[:200], indent=2))
    print(f"\nSaved top 200 to {out}")


if __name__ == "__main__":
    main()
