"""Walk-forward validation to detect overfitting.

Splits historical data into rolling train/test windows and measures
out-of-sample degradation. If in-sample Sharpe >> out-of-sample Sharpe,
the model is overfit to historical data.

Window structure:
  [----TRAIN (70%)----][--TEST (30%)--]
                  [----TRAIN----][--TEST--]
                           [----TRAIN----][--TEST--]

Each window produces in-sample and out-of-sample metrics. The ratio
of OOS/IS performance is the "walk-forward efficiency" — values above
0.5 indicate robust strategies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import sqrt
from typing import Any


@dataclass
class WalkForwardWindow:
    """Results from a single train/test window."""

    window_id: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str

    # In-sample (train) metrics
    is_trades: int = 0
    is_win_rate: float = 0.0
    is_profit_factor: float = 0.0
    is_avg_pnl: float = 0.0
    is_sharpe: float = 0.0

    # Out-of-sample (test) metrics
    oos_trades: int = 0
    oos_win_rate: float = 0.0
    oos_profit_factor: float = 0.0
    oos_avg_pnl: float = 0.0
    oos_sharpe: float = 0.0


@dataclass
class WalkForwardResult:
    """Aggregated walk-forward validation results."""

    windows: list[WalkForwardWindow] = field(default_factory=list)
    total_windows: int = 0
    avg_is_win_rate: float = 0.0
    avg_oos_win_rate: float = 0.0
    avg_is_sharpe: float = 0.0
    avg_oos_sharpe: float = 0.0
    walk_forward_efficiency: float = 0.0  # OOS Sharpe / IS Sharpe
    overfitting_probability: float = 0.0  # P(OOS < 0)
    is_overfit: bool = False
    degradation_pct: float = 0.0  # How much WR drops IS → OOS
    notes: list[str] = field(default_factory=list)


def _compute_window_metrics(
    trades: list[dict[str, Any]],
) -> tuple[int, float, float, float, float]:
    """Compute metrics for a set of trades.

    Returns: (count, win_rate, profit_factor, avg_pnl, sharpe_approx)
    """
    if not trades:
        return 0, 0.0, 0.0, 0.0, 0.0

    count = len(trades)
    wins = sum(1 for t in trades if t.get("pnl_pct", 0) > 10)
    win_rate = wins / count if count > 0 else 0.0

    gross_profit = sum(t["pnl_pct"] for t in trades if t["pnl_pct"] > 0)
    gross_loss = abs(sum(t["pnl_pct"] for t in trades if t["pnl_pct"] < 0))
    pf = gross_profit / gross_loss if gross_loss > 0 else 0.0

    avg_pnl = sum(t["pnl_pct"] for t in trades) / count if count > 0 else 0.0

    # Approximate Sharpe from trade PnLs
    pnls = [t["pnl_pct"] for t in trades]
    if len(pnls) > 1:
        mean_p = sum(pnls) / len(pnls)
        var_p = sum((p - mean_p) ** 2 for p in pnls) / (len(pnls) - 1)
        std_p = sqrt(var_p) if var_p > 0 else 0.01
        sharpe = mean_p / std_p * sqrt(252 / max(7, len(pnls)))
    else:
        sharpe = 0.0

    return count, round(win_rate, 3), round(pf, 2), round(avg_pnl, 2), round(sharpe, 3)


def run_walk_forward(
    trades: list[dict[str, Any]],
    n_windows: int = 5,
    train_pct: float = 0.70,
) -> WalkForwardResult:
    """Run walk-forward validation on backtest trades.

    Splits trades chronologically into overlapping train/test windows
    and compares in-sample vs out-of-sample performance.

    Args:
        trades: List of trade dicts with at minimum 'entry_date' and 'pnl_pct'.
        n_windows: Number of rolling windows.
        train_pct: Fraction of each window used for training.

    Returns:
        WalkForwardResult with per-window and aggregate metrics.
    """
    if len(trades) < 20:
        return WalkForwardResult(
            notes=["Insufficient trades for walk-forward validation (need 20+)"],
        )

    # Sort by entry date
    sorted_trades = sorted(trades, key=lambda t: t.get("entry_date", ""))
    total = len(sorted_trades)

    # Calculate window size and step
    window_size = total  # Each window sees all data
    step = max(1, total // (n_windows + 1))

    windows: list[WalkForwardWindow] = []

    for i in range(n_windows):
        start_idx = i * step
        end_idx = min(start_idx + window_size, total)
        if end_idx - start_idx < 10:
            continue

        split_idx = start_idx + int((end_idx - start_idx) * train_pct)

        train_trades = sorted_trades[start_idx:split_idx]
        test_trades = sorted_trades[split_idx:end_idx]

        if len(train_trades) < 5 or len(test_trades) < 3:
            continue

        is_n, is_wr, is_pf, is_avg, is_sh = _compute_window_metrics(train_trades)
        oos_n, oos_wr, oos_pf, oos_avg, oos_sh = _compute_window_metrics(test_trades)

        window = WalkForwardWindow(
            window_id=i + 1,
            train_start=str(train_trades[0].get("entry_date", "")),
            train_end=str(train_trades[-1].get("entry_date", "")),
            test_start=str(test_trades[0].get("entry_date", "")),
            test_end=str(test_trades[-1].get("entry_date", "")),
            is_trades=is_n,
            is_win_rate=is_wr,
            is_profit_factor=is_pf,
            is_avg_pnl=is_avg,
            is_sharpe=is_sh,
            oos_trades=oos_n,
            oos_win_rate=oos_wr,
            oos_profit_factor=oos_pf,
            oos_avg_pnl=oos_avg,
            oos_sharpe=oos_sh,
        )
        windows.append(window)

    if not windows:
        return WalkForwardResult(notes=["No valid windows generated"])

    # Aggregate
    n_w = len(windows)
    avg_is_wr = sum(w.is_win_rate for w in windows) / n_w
    avg_oos_wr = sum(w.oos_win_rate for w in windows) / n_w
    avg_is_sh = sum(w.is_sharpe for w in windows) / n_w
    avg_oos_sh = sum(w.oos_sharpe for w in windows) / n_w

    wfe = avg_oos_sh / avg_is_sh if avg_is_sh > 0 else 0.0
    overfit_prob = sum(1 for w in windows if w.oos_sharpe < 0) / n_w
    degradation = (avg_is_wr - avg_oos_wr) / avg_is_wr * 100 if avg_is_wr > 0 else 0.0

    notes: list[str] = []
    is_overfit = False

    if wfe < 0.3:
        notes.append("SEVERE OVERFITTING: Walk-forward efficiency < 30%")
        is_overfit = True
    elif wfe < 0.5:
        notes.append("MODERATE OVERFITTING: Walk-forward efficiency < 50%")
        is_overfit = True
    else:
        notes.append(f"Strategy appears robust (WFE={wfe:.1%})")

    if degradation > 30:
        notes.append(f"Win rate degrades {degradation:.0f}% out-of-sample")
    if overfit_prob > 0.5:
        notes.append(
            f"Over half of OOS windows are unprofitable ({overfit_prob:.0%})"
        )

    return WalkForwardResult(
        windows=windows,
        total_windows=n_w,
        avg_is_win_rate=round(avg_is_wr, 3),
        avg_oos_win_rate=round(avg_oos_wr, 3),
        avg_is_sharpe=round(avg_is_sh, 3),
        avg_oos_sharpe=round(avg_oos_sh, 3),
        walk_forward_efficiency=round(wfe, 3),
        overfitting_probability=round(overfit_prob, 3),
        is_overfit=is_overfit,
        degradation_pct=round(degradation, 1),
        notes=notes,
    )
