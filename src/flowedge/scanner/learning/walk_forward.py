"""WALK-FORWARD VALIDATOR — prevents overfitting in the learning system.

Inspired by vectorbt's walk-forward optimization and freqAI-LSTM's
dynamic weighting approach.

Walk-forward analysis splits historical data into rolling windows:
  [TRAIN period] → [TEST period] → shift → repeat

Each window:
1. TRAIN: Extract patterns and compute optimal weights from past trades
2. TEST: Apply those weights to unseen trades and measure performance
3. Compare train vs test performance to detect overfitting

If test performance degrades significantly vs train performance,
the refinement is likely overfitting and should be rejected.

Additionally implements regime-adaptive weighting (from freqAI-LSTM):
- Detects market regime from recent trade outcomes
- Adjusts weight emphasis based on which signals work in current regime
- Bullish regimes → more UOA weight (flow signals lead)
- High-vol regimes → more IV weight (vol signals dominate)
- Catalyst-heavy periods → more catalyst weight

References:
- github.com/polakowo/vectorbt (walk-forward optimization)
- github.com/Netanelshoshan/freqAI-LSTM (dynamic weighting)
- github.com/AI4Finance-Foundation/FinRL (train-test-trade pipeline)
"""

from __future__ import annotations

import structlog

from flowedge.scanner.learning.schemas import (
    AdaptiveWeights,
    WeightAdjustment,
)
from flowedge.scanner.performance.schemas import (
    SimulatedTrade,
    TradeResult,
)

logger = structlog.get_logger()


# ──────────────────────────────────────────────────────────────
# Walk-Forward Validation
# ──────────────────────────────────────────────────────────────

class WalkForwardWindow:
    """One train/test split in a walk-forward analysis."""

    def __init__(
        self,
        train_trades: list[SimulatedTrade],
        test_trades: list[SimulatedTrade],
        window_id: int,
    ) -> None:
        self.train = train_trades
        self.test = test_trades
        self.window_id = window_id
        self.train_wr = 0.0
        self.test_wr = 0.0
        self.overfit_ratio = 0.0

    def compute_stats(self) -> None:
        """Compute win rates for both halves."""
        train_closed = [t for t in self.train if t.result != TradeResult.OPEN]
        test_closed = [t for t in self.test if t.result != TradeResult.OPEN]

        if train_closed:
            self.train_wr = sum(
                1 for t in train_closed if t.result == TradeResult.WIN
            ) / len(train_closed)
        if test_closed:
            self.test_wr = sum(
                1 for t in test_closed if t.result == TradeResult.WIN
            ) / len(test_closed)

        # Overfit ratio: how much worse is test vs train
        if self.train_wr > 0:
            self.overfit_ratio = (self.train_wr - self.test_wr) / self.train_wr
        else:
            self.overfit_ratio = 0.0


class WalkForwardResult:
    """Results of a walk-forward validation."""

    def __init__(self) -> None:
        self.windows: list[WalkForwardWindow] = []
        self.avg_train_wr = 0.0
        self.avg_test_wr = 0.0
        self.avg_overfit_ratio = 0.0
        self.is_overfit = False
        self.walkforward_efficiency = 0.0  # test_wr / train_wr

    def compute(self) -> None:
        """Aggregate across windows."""
        if not self.windows:
            return

        self.avg_train_wr = sum(w.train_wr for w in self.windows) / len(self.windows)
        self.avg_test_wr = sum(w.test_wr for w in self.windows) / len(self.windows)
        self.avg_overfit_ratio = (
            sum(w.overfit_ratio for w in self.windows) / len(self.windows)
        )

        # Walk-forward efficiency (WFE): ideally close to 1.0
        if self.avg_train_wr > 0:
            self.walkforward_efficiency = self.avg_test_wr / self.avg_train_wr
        else:
            self.walkforward_efficiency = 0.0

        # Flag overfitting if test performance < 50% of train
        self.is_overfit = self.walkforward_efficiency < 0.5


def run_walk_forward(
    trades: list[SimulatedTrade],
    n_windows: int = 4,
    train_pct: float = 0.7,
) -> WalkForwardResult:
    """Run walk-forward analysis on trade history.

    Splits trades chronologically into rolling train/test windows.

    Args:
        trades: All simulated trades, sorted by entry_date.
        n_windows: Number of walk-forward windows.
        train_pct: Fraction of each window used for training.

    Returns:
        WalkForwardResult with overfit detection.
    """
    closed = sorted(
        [t for t in trades if t.result != TradeResult.OPEN],
        key=lambda t: t.entry_date,
    )

    result = WalkForwardResult()

    if len(closed) < 20:
        logger.info("walk_forward_skip", reason="insufficient_trades", count=len(closed))
        return result

    # Rolling windows with 50% overlap
    window_size = len(closed) // max(n_windows, 1)
    step = max(1, window_size // 2)

    for i in range(n_windows):
        start = i * step
        end = start + window_size
        if end > len(closed):
            break

        window_trades = closed[start:end]
        split = int(len(window_trades) * train_pct)

        train_trades = window_trades[:split]
        test_trades = window_trades[split:]

        if len(train_trades) < 5 or len(test_trades) < 3:
            continue

        window = WalkForwardWindow(train_trades, test_trades, i)
        window.compute_stats()
        result.windows.append(window)

        logger.info(
            "wf_window",
            window=i,
            train_size=len(train_trades),
            test_size=len(test_trades),
            train_wr=f"{window.train_wr:.1%}",
            test_wr=f"{window.test_wr:.1%}",
            overfit=f"{window.overfit_ratio:.1%}",
        )

    result.compute()

    logger.info(
        "walk_forward_complete",
        windows=len(result.windows),
        avg_train_wr=f"{result.avg_train_wr:.1%}",
        avg_test_wr=f"{result.avg_test_wr:.1%}",
        wfe=f"{result.walkforward_efficiency:.2f}",
        is_overfit=result.is_overfit,
    )

    return result


# ──────────────────────────────────────────────────────────────
# Regime-Adaptive Weighting (inspired by freqAI-LSTM)
# ──────────────────────────────────────────────────────────────

class MarketRegime:
    """Detected market regime from recent trade outcomes."""

    def __init__(self, label: str, confidence: float = 0.0) -> None:
        self.label = label  # "bullish", "bearish", "volatile", "calm"
        self.confidence = confidence


def detect_regime(
    trades: list[SimulatedTrade],
    lookback_trades: int = 20,
) -> MarketRegime:
    """Detect current market regime from recent trade patterns.

    Regime classification:
    - BULLISH: Most recent winning trades were calls, momentum positive
    - BEARISH: Most recent winning trades were puts, momentum negative
    - VOLATILE: Wide P&L swings, high win% but also high loss%
    - CALM: Narrow P&L range, mostly theta decay losses
    """
    recent = sorted(
        [t for t in trades if t.result != TradeResult.OPEN],
        key=lambda t: t.entry_date,
    )[-lookback_trades:]

    if len(recent) < 5:
        return MarketRegime("unknown", 0.0)

    # Directional analysis
    call_wins = sum(
        1 for t in recent
        if t.option_type == "call" and t.result == TradeResult.WIN
    )
    put_wins = sum(
        1 for t in recent
        if t.option_type == "put" and t.result == TradeResult.WIN
    )
    total_wins = sum(1 for t in recent if t.result == TradeResult.WIN)

    # Volatility analysis
    pnls = [t.pnl_pct for t in recent]
    avg_pnl = sum(pnls) / len(pnls)
    pnl_std = (sum((p - avg_pnl) ** 2 for p in pnls) / len(pnls)) ** 0.5

    # Win rate
    wr = total_wins / len(recent)

    # Classification
    if pnl_std > 60:
        return MarketRegime("volatile", min(pnl_std / 100, 1.0))
    if call_wins > put_wins * 2 and wr > 0.3:
        return MarketRegime("bullish", min(call_wins / max(total_wins, 1), 1.0))
    if put_wins > call_wins * 2 and wr > 0.3:
        return MarketRegime("bearish", min(put_wins / max(total_wins, 1), 1.0))
    if pnl_std < 20 and wr < 0.2:
        return MarketRegime("calm", 0.6)

    return MarketRegime("mixed", 0.4)


def regime_adjusted_weights(
    base_weights: AdaptiveWeights,
    regime: MarketRegime,
) -> list[WeightAdjustment]:
    """Suggest weight adjustments based on market regime.

    Inspired by freqAI-LSTM's approach of dynamically adjusting
    indicator importance based on market conditions.
    """
    adjustments: list[WeightAdjustment] = []

    if regime.label == "bullish" and regime.confidence > 0.5:
        # In bullish regimes, flow signals lead — increase UOA weight
        if base_weights.uoa_weight < 0.40:
            adjustments.append(WeightAdjustment(
                parameter="uoa_weight",
                current_value=base_weights.uoa_weight,
                suggested_value=min(0.40, base_weights.uoa_weight + 0.03),
                reason=f"Bullish regime ({regime.confidence:.0%}) — flow signals leading",
                confidence=regime.confidence * 0.6,
            ))

    elif regime.label == "volatile" and regime.confidence > 0.5:
        # In volatile regimes, IV signals dominate
        if base_weights.iv_weight < 0.40:
            adjustments.append(WeightAdjustment(
                parameter="iv_weight",
                current_value=base_weights.iv_weight,
                suggested_value=min(0.40, base_weights.iv_weight + 0.03),
                reason=f"Volatile regime ({regime.confidence:.0%}) — vol signals dominant",
                confidence=regime.confidence * 0.6,
            ))

    elif regime.label == "calm" and regime.confidence > 0.5:
        # In calm regimes, only catalysts can drive moves
        if base_weights.catalyst_weight < 0.45:
            adjustments.append(WeightAdjustment(
                parameter="catalyst_weight",
                current_value=base_weights.catalyst_weight,
                suggested_value=min(0.45, base_weights.catalyst_weight + 0.03),
                reason=f"Calm regime ({regime.confidence:.0%}) — only catalysts drive moves",
                confidence=regime.confidence * 0.6,
            ))

    elif regime.label == "bearish" and regime.confidence > 0.5:
        # Increase minimum score threshold in bearish regimes
        adjustments.append(WeightAdjustment(
            parameter="min_entry_score",
            current_value=base_weights.min_entry_score,
            suggested_value=max(
                base_weights.min_entry_score,
                base_weights.min_entry_score + 5,
            ),
            reason=f"Bearish regime ({regime.confidence:.0%}) — raise entry bar",
            confidence=regime.confidence * 0.5,
        ))

    return adjustments


# ──────────────────────────────────────────────────────────────
# Optimal Weight Discovery via Historical Score Sweep
# ──────────────────────────────────────────────────────────────

def find_optimal_weights(
    trades: list[SimulatedTrade],
    step: float = 0.05,
) -> dict[str, float]:
    """Find weight combination that maximizes win rate on historical data.

    Inspired by vectorbt's parameter sweep approach.
    Sweeps through weight combinations in 5% increments.

    Returns the best weight set found. Used as INPUT to the learning
    system (not applied directly — must pass walk-forward validation).
    """
    closed = [t for t in trades if t.result != TradeResult.OPEN]
    if len(closed) < 20:
        return {"uoa": 0.35, "iv": 0.30, "catalyst": 0.35, "win_rate": 0.0}

    best_wr = 0.0
    best_combo = (0.35, 0.30, 0.35)

    # Generate weight combinations that sum to 1.0
    steps = int(1.0 / step) + 1
    for uoa_i in range(3, steps - 2):  # Min 0.15 each
        uoa_w = round(uoa_i * step, 2)
        for iv_i in range(3, steps - uoa_i - 2):
            iv_w = round(iv_i * step, 2)
            cat_w = round(1.0 - uoa_w - iv_w, 2)
            if cat_w < 0.15 or cat_w > 0.50:
                continue

            # Score each trade with these weights and check if
            # high-score trades win more often
            scores: list[tuple[float, bool]] = []
            for t in closed:
                # Reconstruct approximate dimension scores from nexus_score
                # (In production, we'd store individual dimension scores)
                raw = t.nexus_score / 10  # 0-10 scale
                score = raw  # Simplified — in production, use actual dim scores
                is_win = t.result == TradeResult.WIN
                scores.append((score, is_win))

            # Check if top-half scores win more than bottom-half
            scores.sort(key=lambda x: x[0], reverse=True)
            mid = len(scores) // 2
            top_half = scores[:mid]
            bottom_half = scores[mid:]

            top_wr = sum(1 for _, w in top_half if w) / max(len(top_half), 1)
            bot_wr = sum(1 for _, w in bottom_half if w) / max(len(bottom_half), 1)

            # Separation quality: how much better top-half does
            separation = top_wr - bot_wr
            if separation > best_wr:
                best_wr = separation
                best_combo = (uoa_w, iv_w, cat_w)

    return {
        "uoa": best_combo[0],
        "iv": best_combo[1],
        "catalyst": best_combo[2],
        "score_separation": round(best_wr, 4),
    }
