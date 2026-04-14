"""Ticker Universe Curator — analyzes per-ticker performance and curates the universe.

Evaluates each ticker's contribution to overall portfolio quality, identifies
underperformers to remove, and suggests candidates from the full 33-ticker
universe that may improve diversification and returns.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

from flowedge.council.models import (
    Finding,
    Recommendation,
    RecommendationType,
    Severity,
    SpecialistReview,
    TickerScorecard,
)
from flowedge.council.specialists.base import BaseSpecialist
from flowedge.scanner.backtest.scalp_config import ALL_33_TICKERS, ScalpConfig
from flowedge.scanner.backtest.schemas import BacktestResult, TradeOutcome

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

@dataclass
class _TickerStats:
    """Accumulated stats for a single ticker."""

    ticker: str
    trades: int = 0
    wins: int = 0
    losses: int = 0
    total_pnl_pct: float = 0.0
    best_trade_pct: float = float("-inf")
    worst_trade_pct: float = float("inf")
    pnl_pcts: list[float] = field(default_factory=list)
    hold_days: list[int] = field(default_factory=list)

    @property
    def win_rate(self) -> float:
        return self.wins / self.trades if self.trades > 0 else 0.0

    @property
    def avg_pnl_pct(self) -> float:
        return self.total_pnl_pct / self.trades if self.trades > 0 else 0.0

    @property
    def avg_hold_days(self) -> float:
        return sum(self.hold_days) / len(self.hold_days) if self.hold_days else 0.0


def _compute_ticker_stats(result: BacktestResult) -> dict[str, _TickerStats]:
    """Build per-ticker stats from individual trades."""
    stats: dict[str, _TickerStats] = {}

    for trade in result.trades:
        t = trade.ticker
        if t not in stats:
            stats[t] = _TickerStats(ticker=t)
        s = stats[t]
        s.trades += 1
        s.pnl_pcts.append(trade.pnl_pct)
        s.total_pnl_pct += trade.pnl_pct
        s.hold_days.append(trade.hold_days)

        if trade.outcome == TradeOutcome.WIN:
            s.wins += 1
        elif trade.outcome == TradeOutcome.LOSS:
            s.losses += 1

        if trade.pnl_pct > s.best_trade_pct:
            s.best_trade_pct = trade.pnl_pct
        if trade.pnl_pct < s.worst_trade_pct:
            s.worst_trade_pct = trade.pnl_pct

    # Fix sentinel values for tickers that had trades
    for s in stats.values():
        if s.best_trade_pct == float("-inf"):
            s.best_trade_pct = 0.0
        if s.worst_trade_pct == float("inf"):
            s.worst_trade_pct = 0.0

    return stats


def _composite_score(
    win_rate: float,
    avg_pnl_pct: float,
    trade_count: int,
    max_pnl: float,
    max_trades: int,
) -> float:
    """Composite score: WR * 0.4 + normalized_pnl * 0.3 + trade_count * 0.3.

    PnL and trade count are normalized to [0, 1] relative to the best ticker
    in the current universe. Win rate is already 0-1.
    """
    norm_pnl = (avg_pnl_pct / max_pnl) if max_pnl > 0 else 0.0
    norm_pnl = max(0.0, min(1.0, norm_pnl))  # clamp

    norm_trades = (trade_count / max_trades) if max_trades > 0 else 0.0
    norm_trades = max(0.0, min(1.0, norm_trades))

    return round(win_rate * 0.4 + norm_pnl * 0.3 + norm_trades * 0.3, 4)


# ---------------------------------------------------------------------------
# Specialist
# ---------------------------------------------------------------------------

class TickerCurator(BaseSpecialist):
    """Analyzes per-ticker performance and curates the ticker universe."""

    name: str = "Ticker Universe Curator"
    specialist_id: str = "ticker_curator"

    # Thresholds
    _MIN_TRADES_FOR_JUDGMENT: int = 3
    _UNDERPERFORM_WR: float = 0.50

    def analyze(
        self,
        result: BacktestResult,
        config: ScalpConfig,
        review_date: date,
        history: list[BacktestResult],
    ) -> SpecialistReview:
        findings: list[Finding] = []
        recommendations: list[Recommendation] = []
        metrics: dict[str, float | str] = {}

        overall_wr = result.win_rate
        current_tickers = set(config.tickers)
        all_universe = set(ALL_33_TICKERS)

        # ------------------------------------------------------------------
        # 1. Compute per-ticker stats from trade-level data
        # ------------------------------------------------------------------
        stats = _compute_ticker_stats(result)

        # Also include tickers in config that had zero trades
        for t in current_tickers:
            if t not in stats:
                stats[t] = _TickerStats(ticker=t)

        # ------------------------------------------------------------------
        # 2. Normalization helpers for composite score
        # ------------------------------------------------------------------
        all_avg_pnls = [s.avg_pnl_pct for s in stats.values() if s.trades > 0]
        all_trade_counts = [s.trades for s in stats.values() if s.trades > 0]
        max_pnl = max(all_avg_pnls) if all_avg_pnls else 1.0
        max_trades = max(all_trade_counts) if all_trade_counts else 1

        # ------------------------------------------------------------------
        # 3. Build scorecards and classify tickers
        # ------------------------------------------------------------------
        scorecards: list[TickerScorecard] = []
        top_performers: list[str] = []
        underperformers: list[str] = []
        zero_trade_tickers: list[str] = []
        total_loss_tickers: list[str] = []
        composite_scores: dict[str, float] = {}

        for ticker in sorted(stats):
            s = stats[ticker]
            in_universe = ticker in current_tickers
            score = _composite_score(s.win_rate, s.avg_pnl_pct, s.trades, max_pnl, max_trades)
            composite_scores[ticker] = score

            # Determine recommendation label
            if s.trades == 0 and in_universe:
                rec_label = "remove"
                zero_trade_tickers.append(ticker)
            elif s.trades >= self._MIN_TRADES_FOR_JUDGMENT and s.win_rate == 0.0:
                rec_label = "remove"
                total_loss_tickers.append(ticker)
            elif (
                s.trades >= self._MIN_TRADES_FOR_JUDGMENT
                and s.win_rate < self._UNDERPERFORM_WR
            ):
                rec_label = "watch"
                underperformers.append(ticker)
            elif (
                s.trades >= self._MIN_TRADES_FOR_JUDGMENT
                and s.win_rate > overall_wr
            ):
                rec_label = "keep"
                top_performers.append(ticker)
            elif s.trades > 0:
                rec_label = "hold"
            else:
                rec_label = "hold"

            scorecards.append(
                TickerScorecard(
                    ticker=ticker,
                    trades=s.trades,
                    wins=s.wins,
                    win_rate=round(s.win_rate, 4),
                    total_pnl=round(s.total_pnl_pct, 4),
                    avg_pnl_pct=round(s.avg_pnl_pct, 4),
                    best_trade_pct=round(s.best_trade_pct, 4),
                    worst_trade_pct=round(s.worst_trade_pct, 4),
                    avg_hold_bars=round(s.avg_hold_days, 2),
                    recommendation=rec_label,
                )
            )

        # ------------------------------------------------------------------
        # 4. Findings: top performers
        # ------------------------------------------------------------------
        if top_performers:
            top_details = "; ".join(
                f"{t} WR={stats[t].win_rate:.0%} ({stats[t].trades} trades, "
                f"avg PnL={stats[t].avg_pnl_pct:+.2%})"
                for t in sorted(top_performers, key=lambda t: -composite_scores[t])
            )
            findings.append(
                Finding(
                    title="Top-performing tickers identified",
                    detail=(
                        f"{len(top_performers)} ticker(s) exceed the overall WR "
                        f"of {overall_wr:.0%} with {self._MIN_TRADES_FOR_JUDGMENT}+ trades: "
                        f"{top_details}"
                    ),
                    severity=Severity.INFO,
                    metric_name="top_performers_count",
                    metric_value=len(top_performers),
                    threshold=(
                        f"WR > {overall_wr:.0%} AND "
                        f"trades >= {self._MIN_TRADES_FOR_JUDGMENT}"
                    ),
                    evidence=[
                        f"{t}: WR={stats[t].win_rate:.0%}, trades={stats[t].trades}, "
                        f"PnL={stats[t].total_pnl_pct:+.2%}"
                        for t in top_performers
                    ],
                )
            )

        # ------------------------------------------------------------------
        # 5. Findings: underperformers
        # ------------------------------------------------------------------
        if underperformers:
            under_details = "; ".join(
                f"{t} WR={stats[t].win_rate:.0%} ({stats[t].trades} trades, "
                f"avg PnL={stats[t].avg_pnl_pct:+.2%})"
                for t in sorted(underperformers, key=lambda t: composite_scores[t])
            )
            findings.append(
                Finding(
                    title="Underperforming tickers detected",
                    detail=(
                        f"{len(underperformers)} ticker(s) have WR below "
                        f"{self._UNDERPERFORM_WR:.0%} with {self._MIN_TRADES_FOR_JUDGMENT}+ "
                        f"trades: {under_details}"
                    ),
                    severity=Severity.WARNING,
                    metric_name="underperformers_count",
                    metric_value=len(underperformers),
                    threshold=f"WR < {self._UNDERPERFORM_WR:.0%}",
                    evidence=[
                        f"{t}: WR={stats[t].win_rate:.0%}, trades={stats[t].trades}, "
                        f"PnL={stats[t].total_pnl_pct:+.2%}"
                        for t in underperformers
                    ],
                )
            )

        # ------------------------------------------------------------------
        # 6. Findings: zero-trade tickers (dead weight)
        # ------------------------------------------------------------------
        if zero_trade_tickers:
            findings.append(
                Finding(
                    title="Tickers with zero trades (no signal activity)",
                    detail=(
                        f"{len(zero_trade_tickers)} ticker(s) in the current universe "
                        f"produced zero trades: {', '.join(sorted(zero_trade_tickers))}. "
                        f"These consume scanning resources without contributing."
                    ),
                    severity=Severity.WARNING,
                    metric_name="zero_trade_tickers",
                    metric_value=len(zero_trade_tickers),
                    threshold=0,
                    evidence=[
                        f"{t}: 0 trades in backtest window" for t in zero_trade_tickers
                    ],
                )
            )

        # ------------------------------------------------------------------
        # 7. Findings: 100% loss rate tickers
        # ------------------------------------------------------------------
        if total_loss_tickers:
            loss_details = "; ".join(
                f"{t} ({stats[t].trades} trades, PnL={stats[t].total_pnl_pct:+.2%})"
                for t in total_loss_tickers
            )
            findings.append(
                Finding(
                    title="Tickers with 100% loss rate",
                    detail=(
                        f"{len(total_loss_tickers)} ticker(s) lost every trade: "
                        f"{loss_details}"
                    ),
                    severity=Severity.CRITICAL,
                    metric_name="total_loss_tickers",
                    metric_value=len(total_loss_tickers),
                    threshold=0,
                    evidence=[
                        f"{t}: 0 wins / {stats[t].trades} trades, "
                        f"total PnL={stats[t].total_pnl_pct:+.2%}"
                        for t in total_loss_tickers
                    ],
                )
            )

        # ------------------------------------------------------------------
        # 8. Recommendations: remove dead weight / total-loss tickers
        # ------------------------------------------------------------------
        for ticker in sorted(set(zero_trade_tickers + total_loss_tickers)):
            if ticker not in current_tickers:
                continue
            s = stats[ticker]
            if s.trades == 0:
                reason = "zero signal activity"
                impact = "Reduces scanning overhead with no loss of edge"
            else:
                reason = f"100% loss rate across {s.trades} trades"
                impact = (
                    f"Eliminates drag of {s.total_pnl_pct:+.2%} cumulative PnL loss"
                )

            recommendations.append(
                Recommendation(
                    title=f"Remove {ticker} from universe",
                    rationale=f"{ticker} shows {reason} — keeping it degrades portfolio quality.",
                    rec_type=RecommendationType.TICKER_REMOVE,
                    priority=2,
                    confidence=0.85 if s.trades >= self._MIN_TRADES_FOR_JUDGMENT else 0.65,
                    current_value=f"in universe ({s.trades} trades)",
                    suggested_value="remove",
                    expected_impact=impact,
                    evidence=[
                        f"WR={s.win_rate:.0%}, trades={s.trades}, PnL={s.total_pnl_pct:+.2%}",
                    ],
                )
            )

        # ------------------------------------------------------------------
        # 9. Recommendations: watch underperformers (not immediate remove)
        # ------------------------------------------------------------------
        for ticker in sorted(underperformers):
            s = stats[ticker]
            recommendations.append(
                Recommendation(
                    title=f"Watch {ticker} — underperforming",
                    rationale=(
                        f"{ticker} WR={s.win_rate:.0%} is below {self._UNDERPERFORM_WR:.0%} "
                        f"threshold. Consider removal if trend continues."
                    ),
                    rec_type=RecommendationType.TICKER_REMOVE,
                    priority=3,
                    confidence=0.55,
                    current_value=f"WR={s.win_rate:.0%} ({s.trades} trades)",
                    suggested_value="monitor for 1-2 more sessions",
                    expected_impact=(
                        f"Potential PnL improvement of {abs(s.total_pnl_pct):.2%} if removed"
                        if s.total_pnl_pct < 0
                        else "Marginal impact — WR may revert to mean"
                    ),
                    evidence=[
                        f"WR={s.win_rate:.0%}, trades={s.trades}, "
                        f"avg PnL={s.avg_pnl_pct:+.2%}, total PnL={s.total_pnl_pct:+.2%}",
                    ],
                )
            )

        # ------------------------------------------------------------------
        # 10. Recommendations: suggest adding tickers from ALL_33 not in universe
        # ------------------------------------------------------------------
        candidates_to_add = sorted(all_universe - current_tickers)
        if candidates_to_add:
            # Limit suggestions to a reasonable batch
            max_suggestions = 5
            suggested = candidates_to_add[:max_suggestions]
            for ticker in suggested:
                recommendations.append(
                    Recommendation(
                        title=f"Consider adding {ticker} to universe",
                        rationale=(
                            f"{ticker} is in the full 33-ticker universe but not in the "
                            f"current {len(current_tickers)}-ticker config. "
                            f"May provide diversification and new signal opportunities."
                        ),
                        rec_type=RecommendationType.TICKER_ADD,
                        priority=4,
                        confidence=0.35,
                        current_value="not in universe",
                        suggested_value="add (requires backtest validation)",
                        expected_impact=(
                            "Unknown — needs backtest on historical "
                            "data before inclusion"
                        ),
                        evidence=[
                            f"{ticker} is part of ALL_33_TICKERS universe",
                            f"Current universe has {len(current_tickers)} tickers",
                        ],
                    )
                )

            if len(candidates_to_add) > max_suggestions:
                findings.append(
                    Finding(
                        title="Additional ticker candidates available",
                        detail=(
                            f"{len(candidates_to_add) - max_suggestions} more tickers from "
                            f"ALL_33_TICKERS are not in the current universe: "
                            f"{', '.join(candidates_to_add[max_suggestions:])}"
                        ),
                        severity=Severity.INFO,
                        metric_name="additional_candidates",
                        metric_value=len(candidates_to_add) - max_suggestions,
                        evidence=[
                            f"Full 33-ticker universe has {len(all_universe)} tickers; "
                            f"current config uses {len(current_tickers)}"
                        ],
                    )
                )

        # ------------------------------------------------------------------
        # 11. Ticker ranking by composite score
        # ------------------------------------------------------------------
        ranked = sorted(
            [(t, composite_scores[t]) for t in stats if stats[t].trades > 0],
            key=lambda x: -x[1],
        )
        if ranked:
            ranking_lines = [
                f"#{i+1} {t}: score={sc:.3f} (WR={stats[t].win_rate:.0%}, "
                f"trades={stats[t].trades}, avg PnL={stats[t].avg_pnl_pct:+.2%})"
                for i, (t, sc) in enumerate(ranked)
            ]
            findings.append(
                Finding(
                    title="Ticker composite ranking",
                    detail=(
                        f"Ranked {len(ranked)} active tickers by composite score "
                        f"(WR*0.4 + norm_PnL*0.3 + norm_trades*0.3)."
                    ),
                    severity=Severity.INFO,
                    metric_name="ranked_tickers",
                    metric_value=len(ranked),
                    evidence=ranking_lines,
                )
            )

        # ------------------------------------------------------------------
        # 12. Health score: universe quality (40%), diversification (30%),
        #     dead weight (30%)
        # ------------------------------------------------------------------
        active_tickers = [
            t for t in current_tickers
            if stats.get(t, _TickerStats(ticker=t)).trades > 0
        ]
        active_count = len(active_tickers)
        universe_size = len(current_tickers)

        # Universe quality: average composite score of active tickers (0-1 -> 0-100)
        if active_tickers:
            avg_composite = sum(composite_scores.get(t, 0) for t in active_tickers) / active_count
            universe_quality = min(100.0, avg_composite * 100)
        else:
            universe_quality = 0.0

        # Diversification: how spread are trades across tickers?
        # Perfect = equal trades per ticker; worst = all trades in 1 ticker
        if active_count > 1 and result.total_trades > 0:
            trade_shares = [
                stats[t].trades / result.total_trades
                for t in active_tickers
            ]
            # Herfindahl-Hirschman Index (lower = more diverse)
            hhi = sum(s ** 2 for s in trade_shares)
            # Normalize: min HHI = 1/N, max HHI = 1.0
            min_hhi = 1.0 / active_count
            if hhi <= min_hhi:
                diversification = 100.0
            else:
                diversification = max(0.0, (1.0 - (hhi - min_hhi) / (1.0 - min_hhi)) * 100)
        elif active_count == 1:
            diversification = 20.0  # single ticker = very low diversification
        else:
            diversification = 0.0

        # Dead weight: penalize for inactive tickers
        if universe_size > 0:
            dead_weight_ratio = len(zero_trade_tickers) / universe_size
            dead_weight_score = (1.0 - dead_weight_ratio) * 100
        else:
            dead_weight_score = 0.0

        health_score = round(
            universe_quality * 0.40
            + diversification * 0.30
            + dead_weight_score * 0.30,
            1,
        )
        health_score = max(0.0, min(100.0, health_score))

        # ------------------------------------------------------------------
        # 13. Severity
        # ------------------------------------------------------------------
        if health_score >= 70:
            severity = Severity.INFO
        elif health_score >= 45:
            severity = Severity.WARNING
        else:
            severity = Severity.CRITICAL

        # ------------------------------------------------------------------
        # 14. Metrics (includes scorecard data)
        # ------------------------------------------------------------------
        metrics["universe_size"] = universe_size
        metrics["active_tickers"] = active_count
        metrics["zero_trade_tickers"] = len(zero_trade_tickers)
        metrics["total_loss_tickers"] = len(total_loss_tickers)
        metrics["top_performers"] = len(top_performers)
        metrics["underperformers"] = len(underperformers)
        metrics["universe_quality_score"] = round(universe_quality, 1)
        metrics["diversification_score"] = round(diversification, 1)
        metrics["dead_weight_score"] = round(dead_weight_score, 1)
        metrics["overall_wr"] = round(overall_wr, 4)
        metrics["candidates_available"] = len(candidates_to_add)

        # Embed top-3 and bottom-3 tickers by composite score
        if ranked:
            for i, (t, sc) in enumerate(ranked[:3]):
                metrics[f"top{i+1}_ticker"] = t
                metrics[f"top{i+1}_score"] = round(sc, 4)
            for i, (t, sc) in enumerate(reversed(ranked[-3:])):
                metrics[f"bottom{i+1}_ticker"] = t
                metrics[f"bottom{i+1}_score"] = round(sc, 4)

        # Embed per-ticker scorecard data into metrics as serializable dict
        metrics["ticker_scorecards_count"] = len(scorecards)

        # ------------------------------------------------------------------
        # 15. Summary
        # ------------------------------------------------------------------
        summary_parts = [
            f"Universe has {universe_size} tickers ({active_count} active, "
            f"{len(zero_trade_tickers)} zero-trade).",
        ]
        if top_performers:
            summary_parts.append(
                f"Top performers: {', '.join(sorted(top_performers)[:3])}."
            )
        if underperformers or total_loss_tickers:
            problem_tickers = sorted(set(underperformers + total_loss_tickers))
            summary_parts.append(
                f"Problem tickers: {', '.join(problem_tickers[:3])}."
            )
        if candidates_to_add:
            summary_parts.append(
                f"{len(candidates_to_add)} candidates available for expansion."
            )

        return SpecialistReview(
            specialist_name=self.name,
            specialist_id=self.specialist_id,
            review_date=review_date,
            health_score=health_score,
            severity=severity,
            summary=" ".join(summary_parts),
            findings=findings,
            recommendations=recommendations,
            metrics=metrics,
        )
