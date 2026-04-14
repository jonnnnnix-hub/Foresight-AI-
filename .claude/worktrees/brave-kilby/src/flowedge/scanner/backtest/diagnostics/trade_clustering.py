"""Trade clustering and correlation detection.

Identifies when losses cluster together (correlated drawdowns),
which usually means the portfolio is over-exposed to a single
factor (e.g., all positions are bullish tech).

Key metrics:
- Loss cluster detection: consecutive losses and their severity
- Ticker concentration: how many trades overlap on same tickers
- Directional concentration: all-bull or all-bear exposure
- Temporal clustering: losses concentrated in specific months
- Drawdown recovery analysis: how long to recover from worst DD
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class LossCluster:
    """A cluster of consecutive losing trades."""

    start_date: str
    end_date: str
    consecutive_losses: int
    total_loss_pct: float
    avg_loss_pct: float
    tickers_involved: list[str] = field(default_factory=list)
    regimes_involved: list[str] = field(default_factory=list)


@dataclass
class ConcentrationMetric:
    """Measures portfolio concentration along a dimension."""

    dimension: str  # "ticker", "direction", "strategy", "regime"
    herfindahl_index: float = 0.0  # 0=diversified, 1=concentrated
    max_concentration: float = 0.0
    max_item: str = ""
    is_over_concentrated: bool = False


@dataclass
class ClusteringResult:
    """Full trade clustering analysis."""

    loss_clusters: list[LossCluster] = field(default_factory=list)
    worst_cluster: LossCluster | None = None
    max_consecutive_losses: int = 0
    avg_cluster_size: float = 0.0

    # Concentration metrics
    ticker_concentration: ConcentrationMetric | None = None
    direction_concentration: ConcentrationMetric | None = None
    strategy_concentration: ConcentrationMetric | None = None

    # Monthly pattern
    worst_month: str = ""
    best_month: str = ""
    monthly_win_rates: dict[str, float] = field(default_factory=dict)

    # Recovery
    max_drawdown_trades: int = 0  # How many trades until DD recovered
    drawdown_recovery_note: str = ""

    notes: list[str] = field(default_factory=list)


def _herfindahl(counts: dict[str, int]) -> float:
    """Compute Herfindahl–Hirschman Index (concentration measure)."""
    total = sum(counts.values())
    if total == 0:
        return 0.0
    shares = [c / total for c in counts.values()]
    return sum(s ** 2 for s in shares)


def _compute_concentration(
    trades: list[dict[str, Any]],
    key: str,
) -> ConcentrationMetric:
    """Compute concentration along a given dimension."""
    counts: dict[str, int] = {}
    for t in trades:
        val = str(t.get(key, "unknown"))
        counts[val] = counts.get(val, 0) + 1

    hhi = _herfindahl(counts)
    max_item = max(counts, key=counts.get, default="")  # type: ignore[arg-type]
    max_conc = counts.get(max_item, 0) / len(trades) if trades else 0.0

    return ConcentrationMetric(
        dimension=key,
        herfindahl_index=round(hhi, 3),
        max_concentration=round(max_conc, 3),
        max_item=max_item,
        is_over_concentrated=hhi > 0.35,  # > 35% HHI is concentrated
    )


def run_clustering_analysis(
    trades: list[dict[str, Any]],
) -> ClusteringResult:
    """Analyze trade clustering, concentration, and correlation patterns.

    Args:
        trades: Sorted list of trade dicts with pnl_pct, ticker,
                direction, strategy, entry_date.

    Returns:
        ClusteringResult with clusters, concentration, and recovery metrics.
    """
    if len(trades) < 5:
        return ClusteringResult(notes=["Insufficient trades for clustering analysis"])

    sorted_trades = sorted(trades, key=lambda t: str(t.get("entry_date", "")))

    # ── 1. Detect loss clusters ──
    clusters: list[LossCluster] = []
    current_cluster: list[dict[str, Any]] = []

    for t in sorted_trades:
        pnl = float(t.get("pnl_pct", 0))
        if pnl < -10:  # Loss
            current_cluster.append(t)
        else:
            if len(current_cluster) >= 3:  # 3+ consecutive losses = cluster
                total_loss = sum(
                    float(x.get("pnl_pct", 0)) for x in current_cluster
                )
                clusters.append(LossCluster(
                    start_date=str(current_cluster[0].get("entry_date", "")),
                    end_date=str(current_cluster[-1].get("entry_date", "")),
                    consecutive_losses=len(current_cluster),
                    total_loss_pct=round(total_loss, 2),
                    avg_loss_pct=round(total_loss / len(current_cluster), 2),
                    tickers_involved=list({
                        str(x.get("ticker", "")) for x in current_cluster
                    }),
                    regimes_involved=list({
                        str(x.get("regime", "")) for x in current_cluster
                    }),
                ))
            current_cluster = []

    # Don't forget trailing cluster
    if len(current_cluster) >= 3:
        total_loss = sum(float(x.get("pnl_pct", 0)) for x in current_cluster)
        clusters.append(LossCluster(
            start_date=str(current_cluster[0].get("entry_date", "")),
            end_date=str(current_cluster[-1].get("entry_date", "")),
            consecutive_losses=len(current_cluster),
            total_loss_pct=round(total_loss, 2),
            avg_loss_pct=round(total_loss / len(current_cluster), 2),
            tickers_involved=list({
                str(x.get("ticker", "")) for x in current_cluster
            }),
            regimes_involved=list({
                str(x.get("regime", "")) for x in current_cluster
            }),
        ))

    worst = max(clusters, key=lambda c: c.consecutive_losses, default=None)
    max_consec = worst.consecutive_losses if worst else 0
    avg_size = (
        sum(c.consecutive_losses for c in clusters) / len(clusters)
        if clusters else 0.0
    )

    # ── 2. Concentration metrics ──
    ticker_conc = _compute_concentration(sorted_trades, "ticker")
    dir_conc = _compute_concentration(sorted_trades, "direction")
    strat_conc = _compute_concentration(sorted_trades, "strategy")

    # ── 3. Monthly pattern ──
    monthly_trades: dict[str, list[float]] = {}
    for t in sorted_trades:
        entry = str(t.get("entry_date", ""))
        if len(entry) >= 7:
            month = entry[:7]
            monthly_trades.setdefault(month, []).append(
                float(t.get("pnl_pct", 0))
            )

    monthly_wr: dict[str, float] = {}
    for month, pnls in monthly_trades.items():
        wins = sum(1 for p in pnls if p > 10)
        monthly_wr[month] = round(wins / len(pnls), 3) if pnls else 0.0

    best_month = max(monthly_wr, key=monthly_wr.get, default="")  # type: ignore[arg-type]
    worst_month = min(monthly_wr, key=monthly_wr.get, default="")  # type: ignore[arg-type]

    # ── 4. Drawdown recovery ──
    cumulative = 0.0
    peak = 0.0
    in_dd = False
    dd_start = 0
    max_dd_trades = 0

    for i, t in enumerate(sorted_trades):
        cumulative += float(t.get("pnl_pct", 0))
        if cumulative > peak:
            if in_dd:
                recovery_len = i - dd_start
                max_dd_trades = max(max_dd_trades, recovery_len)
                in_dd = False
            peak = cumulative
        elif not in_dd and cumulative < peak - 10:
            in_dd = True
            dd_start = i

    # ── 5. Notes ──
    notes: list[str] = []
    if clusters:
        notes.append(
            f"Found {len(clusters)} loss clusters "
            f"(avg size={avg_size:.1f} consecutive losses)"
        )
    if worst:
        notes.append(
            f"Worst cluster: {worst.consecutive_losses} losses, "
            f"{worst.total_loss_pct:+.0f}% total "
            f"({', '.join(worst.tickers_involved[:3])})"
        )
    if ticker_conc.is_over_concentrated:
        notes.append(
            f"TICKER CONCENTRATION: {ticker_conc.max_item} has "
            f"{ticker_conc.max_concentration:.0%} of trades"
        )
    if dir_conc.is_over_concentrated:
        notes.append(
            f"DIRECTION CONCENTRATION: {dir_conc.max_item} has "
            f"{dir_conc.max_concentration:.0%} of trades"
        )
    if max_dd_trades > 0:
        notes.append(
            f"Longest drawdown recovery: {max_dd_trades} trades"
        )

    return ClusteringResult(
        loss_clusters=clusters,
        worst_cluster=worst,
        max_consecutive_losses=max_consec,
        avg_cluster_size=round(avg_size, 1),
        ticker_concentration=ticker_conc,
        direction_concentration=dir_conc,
        strategy_concentration=strat_conc,
        worst_month=worst_month,
        best_month=best_month,
        monthly_win_rates=monthly_wr,
        max_drawdown_trades=max_dd_trades,
        notes=notes,
    )
