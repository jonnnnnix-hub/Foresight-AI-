"""Signal Quality Analyst — evaluates entry filter effectiveness.

Parses filter rejection stats from backtest notes, analyzes win-rate
by conviction/signal-score buckets, and grades overall signal quality.
Produces actionable recommendations when filters are too tight (starving
the model of trades) or too loose (diluting win rate).
"""

from __future__ import annotations

import json
import logging
from datetime import date

from flowedge.council.models import (
    Finding,
    Recommendation,
    RecommendationType,
    Severity,
    SpecialistReview,
)
from flowedge.council.specialists.base import BaseSpecialist
from flowedge.scanner.backtest.scalp_config import ScalpConfig
from flowedge.scanner.backtest.schemas import BacktestResult, TradeOutcome

logger = logging.getLogger(__name__)

# ── Thresholds ────────────────────────────────────────────────────

_MIN_TRADES_HEALTHY = 30          # Below this the model is trade-starved
_MIN_TRADES_WARNING = 15          # Critical starvation threshold
_HIGH_WR_TARGET = 0.70            # Target win rate for the scalp model
_LOW_WR_THRESHOLD = 0.55          # Below this, filters are too loose
_FILTER_DOMINANCE_PCT = 0.40      # If one filter rejects >40% of all rejections
_MIN_SIGNAL_SCORE_FOR_BUCKET = 3  # Trades below this score are "low conviction"


class SignalQualityAnalyst(BaseSpecialist):
    """Specialist that evaluates entry signal quality and filter efficiency."""

    name: str = "Signal Quality Analyst"
    specialist_id: str = "signal_analyst"

    # ── Main analysis ─────────────────────────────────────────────

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

        # 1. Parse filter stats from notes
        filter_stats = self._parse_filter_stats(result.notes)
        total_rejected = sum(filter_stats.values()) if filter_stats else 0
        metrics["total_filter_rejections"] = total_rejected

        # 2. Signal-to-trade conversion
        signals_total, signals_matched = self._parse_signal_counts(result.notes)
        conversion_rate = (
            signals_matched / signals_total if signals_total > 0 else 0.0
        )
        metrics["signals_total"] = signals_total
        metrics["signals_matched"] = signals_matched
        metrics["signal_conversion_rate"] = round(conversion_rate, 4)

        # 3. Win rate by conviction / signal score buckets
        bucket_stats = self._compute_bucket_stats(result)
        for bucket_name, stats in bucket_stats.items():
            metrics[f"bucket_{bucket_name}_trades"] = stats["trades"]
            metrics[f"bucket_{bucket_name}_wr"] = round(stats["win_rate"], 4)

        # 4. Score-bucket analysis from BacktestResult (pre-computed)
        if result.by_score_bucket:
            for bucket_key, bucket_data in result.by_score_bucket.items():
                wr = bucket_data.get("win_rate", 0.0)
                count = bucket_data.get("trades", 0)
                metrics[f"score_bucket_{bucket_key}_wr"] = round(wr, 4)
                metrics[f"score_bucket_{bucket_key}_n"] = count

        # 5. Generate findings
        findings.extend(self._assess_filter_efficiency(filter_stats, total_rejected))
        findings.extend(self._assess_trade_volume(result, config))
        findings.extend(self._assess_win_rate_quality(result, bucket_stats))
        findings.extend(self._assess_conversion_rate(conversion_rate, signals_total))

        # 6. Generate recommendations
        recommendations.extend(
            self._recommend_filter_adjustments(
                filter_stats, total_rejected, result, config, conversion_rate
            )
        )

        # 7. Compute health score
        health_score = self._compute_health_score(
            result, filter_stats, total_rejected, conversion_rate
        )
        metrics["health_score"] = health_score

        # 8. Determine overall severity
        severity = self._overall_severity(findings)

        summary = self._build_summary(result, health_score, filter_stats, conversion_rate)

        return SpecialistReview(
            specialist_name=self.name,
            specialist_id=self.specialist_id,
            review_date=review_date,
            health_score=health_score,
            severity=severity,
            summary=summary,
            findings=findings,
            recommendations=recommendations,
            metrics=metrics,
        )

    # ── Parsing helpers ───────────────────────────────────────────

    @staticmethod
    def _parse_filter_stats(notes: list[str]) -> dict[str, int]:
        """Extract filter_stats={...} from notes list.

        Notes may contain raw JSON strings or key=value pairs like
        ``filter_stats={"filter_ibs": 5, ...}``.
        """
        for note in notes:
            if "filter_stats" not in note:
                continue
            # Try key=value format first: filter_stats={...}
            prefix = "filter_stats="
            idx = note.find(prefix)
            if idx != -1:
                json_str = note[idx + len(prefix) :]
                try:
                    parsed = json.loads(json_str)
                    if isinstance(parsed, dict):
                        return {k: int(v) for k, v in parsed.items()}
                except (json.JSONDecodeError, ValueError):
                    logger.debug("Failed to parse filter_stats JSON: %s", json_str)
            # Try parsing entire note as JSON
            try:
                parsed = json.loads(note)
                if isinstance(parsed, dict) and "filter_stats" in parsed:
                    stats = parsed["filter_stats"]
                    if isinstance(stats, dict):
                        return {k: int(v) for k, v in stats.items()}
            except (json.JSONDecodeError, ValueError):
                continue
        return {}

    @staticmethod
    def _parse_signal_counts(notes: list[str]) -> tuple[int, int]:
        """Extract signals_total and signals_matched from notes.

        Returns (total, matched). Falls back to (0, 0) if not found.
        """
        total = 0
        matched = 0
        for note in notes:
            try:
                parsed = json.loads(note) if note.strip().startswith("{") else {}
            except (json.JSONDecodeError, ValueError):
                parsed = {}
            if isinstance(parsed, dict):
                if "signals_total" in parsed:
                    total = int(parsed["signals_total"])
                if "signals_matched" in parsed:
                    matched = int(parsed["signals_matched"])
            # Also handle simple key=value format
            if "signals_total=" in note:
                try:
                    val = note.split("signals_total=")[1].split(",")[0].split()[0]
                    total = int(val)
                except (IndexError, ValueError):
                    pass
            if "signals_matched=" in note:
                try:
                    val = note.split("signals_matched=")[1].split(",")[0].split()[0]
                    matched = int(val)
                except (IndexError, ValueError):
                    pass
        return total, matched

    # ── Bucket analysis ───────────────────────────────────────────

    @staticmethod
    def _compute_bucket_stats(
        result: BacktestResult,
    ) -> dict[str, dict[str, float]]:
        """Group trades into conviction / signal-score buckets and compute WR."""
        buckets: dict[str, list[bool]] = {
            "low": [],    # signal_score < 3
            "mid": [],    # 3 <= signal_score < 7
            "high": [],   # signal_score >= 7
        }
        for trade in result.trades:
            is_win = trade.outcome == TradeOutcome.WIN
            score = trade.signal_score
            if score < _MIN_SIGNAL_SCORE_FOR_BUCKET:
                buckets["low"].append(is_win)
            elif score < 7:
                buckets["mid"].append(is_win)
            else:
                buckets["high"].append(is_win)

        stats: dict[str, dict[str, float]] = {}
        for bucket_name, outcomes in buckets.items():
            n = len(outcomes)
            wr = sum(outcomes) / n if n > 0 else 0.0
            stats[bucket_name] = {"trades": float(n), "win_rate": wr}
        return stats

    # ── Finding generators ────────────────────────────────────────

    @staticmethod
    def _assess_filter_efficiency(
        filter_stats: dict[str, int], total_rejected: int
    ) -> list[Finding]:
        findings: list[Finding] = []
        if not filter_stats:
            findings.append(
                Finding(
                    title="No filter rejection data available",
                    detail=(
                        "Could not locate filter_stats in backtest notes. "
                        "Filter efficiency analysis is unavailable."
                    ),
                    severity=Severity.WARNING,
                    metric_name="filter_stats_available",
                    metric_value="missing",
                )
            )
            return findings

        # Identify dominant filter
        if total_rejected > 0:
            sorted_filters = sorted(
                filter_stats.items(), key=lambda kv: kv[1], reverse=True
            )
            top_filter, top_count = sorted_filters[0]
            dominance = top_count / total_rejected

            evidence = [
                f"{name}: {count} rejections ({count / total_rejected:.0%})"
                for name, count in sorted_filters
            ]

            findings.append(
                Finding(
                    title="Filter rejection breakdown",
                    detail=(
                        f"Total rejections: {total_rejected}. "
                        f"Top rejector: {top_filter} ({top_count}, {dominance:.0%} of all). "
                        f"{'WARNING: single filter dominates rejections.' if dominance > _FILTER_DOMINANCE_PCT else 'Rejections are reasonably distributed.'}"
                    ),
                    severity=(
                        Severity.WARNING
                        if dominance > _FILTER_DOMINANCE_PCT
                        else Severity.INFO
                    ),
                    metric_name="top_filter_dominance",
                    metric_value=round(dominance, 4),
                    threshold=_FILTER_DOMINANCE_PCT,
                    evidence=evidence,
                )
            )

            # Check for filters that never reject (possibly redundant)
            zero_filters = [
                name for name, count in filter_stats.items() if count == 0
            ]
            if zero_filters:
                findings.append(
                    Finding(
                        title="Inactive filters detected",
                        detail=(
                            f"{len(zero_filters)} filter(s) rejected zero candidates: "
                            f"{', '.join(zero_filters)}. These may be redundant or "
                            f"their thresholds may be too permissive."
                        ),
                        severity=Severity.INFO,
                        metric_name="inactive_filter_count",
                        metric_value=len(zero_filters),
                        evidence=[f"{f}: 0 rejections" for f in zero_filters],
                    )
                )
        return findings

    @staticmethod
    def _assess_trade_volume(
        result: BacktestResult, config: ScalpConfig
    ) -> list[Finding]:
        findings: list[Finding] = []
        n = result.total_trades
        num_tickers = len(config.tickers) if config.tickers else 1
        trades_per_ticker = n / num_tickers if num_tickers > 0 else n

        if n < _MIN_TRADES_WARNING:
            findings.append(
                Finding(
                    title="Critically low trade count",
                    detail=(
                        f"Only {n} trades across {num_tickers} tickers "
                        f"({trades_per_ticker:.1f}/ticker). Filters may be excessively "
                        f"tight, starving the model of opportunities. Statistical "
                        f"significance of win rate is questionable."
                    ),
                    severity=Severity.CRITICAL,
                    metric_name="total_trades",
                    metric_value=n,
                    threshold=_MIN_TRADES_WARNING,
                    evidence=[
                        f"total_trades={n}",
                        f"num_tickers={num_tickers}",
                        f"trades_per_ticker={trades_per_ticker:.1f}",
                    ],
                )
            )
        elif n < _MIN_TRADES_HEALTHY:
            findings.append(
                Finding(
                    title="Low trade count",
                    detail=(
                        f"{n} trades across {num_tickers} tickers "
                        f"({trades_per_ticker:.1f}/ticker). Consider relaxing filters "
                        f"to increase sample size while monitoring win rate."
                    ),
                    severity=Severity.WARNING,
                    metric_name="total_trades",
                    metric_value=n,
                    threshold=_MIN_TRADES_HEALTHY,
                    evidence=[
                        f"total_trades={n}",
                        f"trades_per_ticker={trades_per_ticker:.1f}",
                    ],
                )
            )
        return findings

    @staticmethod
    def _assess_win_rate_quality(
        result: BacktestResult,
        bucket_stats: dict[str, dict[str, float]],
    ) -> list[Finding]:
        findings: list[Finding] = []
        wr = result.win_rate

        if result.total_trades == 0:
            return findings

        if wr < _LOW_WR_THRESHOLD:
            findings.append(
                Finding(
                    title="Win rate below acceptable threshold",
                    detail=(
                        f"Overall WR {wr:.1%} is below the {_LOW_WR_THRESHOLD:.0%} "
                        f"floor. Filters are not selective enough to maintain edge."
                    ),
                    severity=Severity.CRITICAL,
                    metric_name="win_rate",
                    metric_value=round(wr, 4),
                    threshold=_LOW_WR_THRESHOLD,
                    evidence=[
                        f"win_rate={wr:.4f}",
                        f"wins={result.wins}",
                        f"losses={result.losses}",
                    ],
                )
            )
        elif wr >= _HIGH_WR_TARGET:
            findings.append(
                Finding(
                    title="Win rate meets target",
                    detail=(
                        f"WR {wr:.1%} meets or exceeds the {_HIGH_WR_TARGET:.0%} target. "
                        f"Signal quality is strong."
                    ),
                    severity=Severity.INFO,
                    metric_name="win_rate",
                    metric_value=round(wr, 4),
                    threshold=_HIGH_WR_TARGET,
                    evidence=[f"win_rate={wr:.4f}", f"profit_factor={result.profit_factor:.2f}"],
                )
            )

        # Check if high-conviction trades outperform low-conviction
        low = bucket_stats.get("low", {})
        high = bucket_stats.get("high", {})
        low_n = low.get("trades", 0)
        high_n = high.get("trades", 0)
        if low_n >= 5 and high_n >= 5:
            low_wr = low.get("win_rate", 0)
            high_wr = high.get("win_rate", 0)
            if high_wr <= low_wr:
                findings.append(
                    Finding(
                        title="High-conviction trades do not outperform low-conviction",
                        detail=(
                            f"High-score bucket WR ({high_wr:.1%}, n={int(high_n)}) does not "
                            f"exceed low-score bucket WR ({low_wr:.1%}, n={int(low_n)}). "
                            f"Signal scoring may not be discriminating effectively."
                        ),
                        severity=Severity.WARNING,
                        metric_name="conviction_spread",
                        metric_value=round(high_wr - low_wr, 4),
                        threshold=0.0,
                        evidence=[
                            f"high_bucket_wr={high_wr:.4f} (n={int(high_n)})",
                            f"low_bucket_wr={low_wr:.4f} (n={int(low_n)})",
                        ],
                    )
                )

        return findings

    @staticmethod
    def _assess_conversion_rate(
        conversion_rate: float, signals_total: int
    ) -> list[Finding]:
        findings: list[Finding] = []
        if signals_total == 0:
            return findings

        if conversion_rate < 0.05:
            findings.append(
                Finding(
                    title="Extremely low signal conversion rate",
                    detail=(
                        f"Only {conversion_rate:.1%} of scanned signals converted to trades. "
                        f"The filter stack is rejecting >95% of candidates. "
                        f"This may indicate over-fitting or excessively tight entry criteria."
                    ),
                    severity=Severity.WARNING,
                    metric_name="signal_conversion_rate",
                    metric_value=round(conversion_rate, 4),
                    threshold=0.05,
                    evidence=[f"signals_total={signals_total}", f"conversion={conversion_rate:.4f}"],
                )
            )
        elif conversion_rate > 0.30:
            findings.append(
                Finding(
                    title="High signal conversion rate",
                    detail=(
                        f"{conversion_rate:.1%} of signals convert to trades. "
                        f"Filters may be too permissive if win rate is below target."
                    ),
                    severity=Severity.INFO,
                    metric_name="signal_conversion_rate",
                    metric_value=round(conversion_rate, 4),
                    threshold=0.30,
                    evidence=[f"signals_total={signals_total}", f"conversion={conversion_rate:.4f}"],
                )
            )
        return findings

    # ── Recommendation generators ─────────────────────────────────

    @staticmethod
    def _recommend_filter_adjustments(
        filter_stats: dict[str, int],
        total_rejected: int,
        result: BacktestResult,
        config: ScalpConfig,
        conversion_rate: float,
    ) -> list[Recommendation]:
        recs: list[Recommendation] = []

        # Too few trades + dominant filter => recommend relaxing that filter
        if (
            filter_stats
            and total_rejected > 0
            and result.total_trades < _MIN_TRADES_HEALTHY
        ):
            sorted_filters = sorted(
                filter_stats.items(), key=lambda kv: kv[1], reverse=True
            )
            top_filter, top_count = sorted_filters[0]
            dominance = top_count / total_rejected

            if dominance > _FILTER_DOMINANCE_PCT:
                # Map filter names to config params with suggested relaxation
                param_map: dict[str, tuple[str, str, str]] = {
                    "filter_ibs": (
                        "ibs_threshold",
                        str(config.ibs_threshold),
                        str(round(config.ibs_threshold + 0.05, 2)),
                    ),
                    "filter_rsi3": (
                        "rsi3_threshold",
                        str(config.rsi3_threshold),
                        str(round(config.rsi3_threshold + 5, 1)),
                    ),
                    "filter_vol_spike": (
                        "vol_spike",
                        str(config.vol_spike),
                        str(round(config.vol_spike - 0.5, 1)),
                    ),
                    "filter_drop": (
                        "intraday_drop",
                        str(config.intraday_drop),
                        str(round(config.intraday_drop - 0.001, 4)),
                    ),
                    "filter_low_premium": (
                        "min_premium",
                        str(config.min_premium),
                        str(round(max(0.10, config.min_premium - 0.10), 2)),
                    ),
                }

                if top_filter in param_map:
                    param_name, current, suggested = param_map[top_filter]
                    recs.append(
                        Recommendation(
                            title=f"Relax dominant filter: {top_filter}",
                            rationale=(
                                f"{top_filter} rejects {top_count}/{total_rejected} "
                                f"candidates ({dominance:.0%}). With only "
                                f"{result.total_trades} trades, relaxing this filter "
                                f"should increase trade count while WR is monitored."
                            ),
                            rec_type=RecommendationType.FILTER_CHANGE,
                            priority=2,
                            confidence=0.6,
                            current_value=f"{param_name}={current}",
                            suggested_value=f"{param_name}={suggested}",
                            expected_impact=(
                                "Increase trade count by ~20-40%, "
                                "WR may decrease 2-5pp — monitor closely."
                            ),
                            evidence=[
                                f"{top_filter}_rejections={top_count}",
                                f"total_trades={result.total_trades}",
                                f"dominance={dominance:.2%}",
                            ],
                        )
                    )

        # Win rate too low => tighten filters
        if result.total_trades >= _MIN_TRADES_WARNING and result.win_rate < _LOW_WR_THRESHOLD:
            recs.append(
                Recommendation(
                    title="Tighten entry filters to improve win rate",
                    rationale=(
                        f"WR {result.win_rate:.1%} is below the "
                        f"{_LOW_WR_THRESHOLD:.0%} floor. The model is accepting "
                        f"too many marginal setups."
                    ),
                    rec_type=RecommendationType.FILTER_CHANGE,
                    priority=1,
                    confidence=0.7,
                    current_value=f"win_rate={result.win_rate:.4f}",
                    suggested_value=f"target_wr>={_LOW_WR_THRESHOLD}",
                    expected_impact=(
                        "Fewer trades but higher quality. "
                        "Focus on IBS and RSI3 thresholds first."
                    ),
                    evidence=[
                        f"win_rate={result.win_rate:.4f}",
                        f"total_trades={result.total_trades}",
                        f"profit_factor={result.profit_factor:.2f}",
                    ],
                )
            )

        # Conversion rate extremely low and we have signal counts
        if conversion_rate > 0 and conversion_rate < 0.03:
            recs.append(
                Recommendation(
                    title="Review filter stack — conversion rate critically low",
                    rationale=(
                        f"Only {conversion_rate:.1%} of signals pass all filters. "
                        f"Consider removing or combining redundant filters."
                    ),
                    rec_type=RecommendationType.FILTER_CHANGE,
                    priority=2,
                    confidence=0.5,
                    current_value=f"conversion_rate={conversion_rate:.4f}",
                    suggested_value="conversion_rate>=0.05",
                    expected_impact="More trade candidates without materially hurting selectivity.",
                    evidence=[f"conversion_rate={conversion_rate:.4f}"],
                )
            )

        # If no issues found, recommend holding steady
        if not recs:
            recs.append(
                Recommendation(
                    title="No filter changes recommended",
                    rationale="Signal quality and filter efficiency are within acceptable bounds.",
                    rec_type=RecommendationType.NO_ACTION,
                    priority=5,
                    confidence=0.8,
                    evidence=[
                        f"win_rate={result.win_rate:.4f}",
                        f"total_trades={result.total_trades}",
                    ],
                )
            )

        return recs

    # ── Health score ──────────────────────────────────────────────

    @staticmethod
    def _compute_health_score(
        result: BacktestResult,
        filter_stats: dict[str, int],
        total_rejected: int,
        conversion_rate: float,
    ) -> float:
        """Composite health score 0-100.

        Components:
        - Win-rate contribution (40%): scales linearly from 0 at 40% WR to 100 at 80%+ WR
        - Filter efficiency (30%): penalizes dominant single-filter and inactive filters
        - Trade volume (30%): scales from 0 at 0 trades to 100 at 50+ trades
        """
        # --- Win-rate component (40%) ---
        wr = result.win_rate
        if wr >= 0.80:
            wr_score = 100.0
        elif wr <= 0.40:
            wr_score = 0.0
        else:
            wr_score = (wr - 0.40) / 0.40 * 100.0

        # --- Filter efficiency component (30%) ---
        if not filter_stats or total_rejected == 0:
            # No data available — give neutral score
            filter_score = 50.0
        else:
            sorted_counts = sorted(filter_stats.values(), reverse=True)
            top_dominance = sorted_counts[0] / total_rejected
            zero_count = sum(1 for c in filter_stats.values() if c == 0)
            total_filters = len(filter_stats)

            # Start at 100, penalize for dominance and inactive filters
            filter_score = 100.0
            # Dominance penalty: heavy penalty above 40%
            if top_dominance > _FILTER_DOMINANCE_PCT:
                filter_score -= (top_dominance - _FILTER_DOMINANCE_PCT) * 150
            # Inactive filter penalty: -10 per inactive filter
            if total_filters > 0:
                filter_score -= (zero_count / total_filters) * 30
            filter_score = max(0.0, min(100.0, filter_score))

        # --- Trade volume component (30%) ---
        n = result.total_trades
        if n >= 50:
            vol_score = 100.0
        elif n <= 0:
            vol_score = 0.0
        else:
            vol_score = n / 50.0 * 100.0

        health = 0.40 * wr_score + 0.30 * filter_score + 0.30 * vol_score
        return round(max(0.0, min(100.0, health)), 1)

    # ── Helpers ───────────────────────────────────────────────────

    @staticmethod
    def _overall_severity(findings: list[Finding]) -> Severity:
        """Return the highest severity across all findings."""
        if any(f.severity == Severity.CRITICAL for f in findings):
            return Severity.CRITICAL
        if any(f.severity == Severity.WARNING for f in findings):
            return Severity.WARNING
        return Severity.INFO

    @staticmethod
    def _build_summary(
        result: BacktestResult,
        health_score: float,
        filter_stats: dict[str, int],
        conversion_rate: float,
    ) -> str:
        parts: list[str] = [
            f"Signal health score: {health_score}/100.",
        ]
        if result.total_trades > 0:
            parts.append(
                f"WR {result.win_rate:.1%} across {result.total_trades} trades "
                f"(PF {result.profit_factor:.2f})."
            )
        else:
            parts.append("No trades executed in this period.")

        if filter_stats:
            total = sum(filter_stats.values())
            top_filter = max(filter_stats, key=filter_stats.get)  # type: ignore[arg-type]
            parts.append(
                f"Top rejecting filter: {top_filter} "
                f"({filter_stats[top_filter]}/{total})."
            )
        if conversion_rate > 0:
            parts.append(f"Signal conversion rate: {conversion_rate:.1%}.")

        return " ".join(parts)
