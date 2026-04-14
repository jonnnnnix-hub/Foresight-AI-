"""Five competing scoring models for the tournament engine.

Each model defines a unique scoring philosophy with 8 category weights,
custom rules, and entry logic. They all receive the same Indicators +
regime and return a 0-100 score.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from flowedge.scanner.backtest.strategies import Indicators, MarketRegime
from flowedge.scanner.tournament.schemas import ModelName


class ScoreCategory(StrEnum):
    """The 8 scoring dimensions."""

    TREND_STRUCTURE = "trend_structure"
    MOMENTUM_QUALITY = "momentum_quality"
    VOLUME_FLOW = "volume_flow"
    VOLATILITY_REGIME = "volatility_regime"
    RISK_REWARD = "risk_reward"
    CATALYST_AWARENESS = "catalyst_awareness"
    LIQUIDITY_EXECUTION = "liquidity_execution"
    SENTIMENT_POSITIONING = "sentiment_positioning"


# ── Score Computation from Indicators ──────────────────────────────


def _clamp(value: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, value))


def compute_category_scores(ind: Indicators, regime: MarketRegime) -> dict[str, float]:
    """Compute the 8 category scores (0-100) from available indicators.

    Since we don't have all 8 live data categories in backtest, we derive
    reasonable proxies from the existing technical indicators.
    """
    # 1. trend_structure: SMA spread + ADX → 0-100
    sma_spread = (ind.sma20 - ind.sma50) / ind.sma50 * 100 if ind.sma50 > 0 else 0.0
    # SMA spread of +3% = strong trend → ~70; ADX > 30 adds more
    trend_from_sma = _clamp(50.0 + sma_spread * 10.0)
    trend_from_adx = _clamp(ind.adx14 * 2.0)
    trend_structure = _clamp(trend_from_sma * 0.6 + trend_from_adx * 0.4)

    # 2. momentum_quality: RSI position + momentum_5 + momentum_10
    # RSI near 50-65 = healthy momentum; extremes penalized
    rsi_score: float
    if 45 <= ind.rsi14 <= 65:
        rsi_score = 70.0 + (ind.rsi14 - 45.0)  # 70-90
    elif ind.rsi14 > 65:
        rsi_score = max(30.0, 90.0 - (ind.rsi14 - 65.0) * 2.0)
    else:
        rsi_score = max(20.0, 70.0 - (45.0 - ind.rsi14) * 2.0)
    mom_score = _clamp(50.0 + ind.momentum_5 * 5.0 + ind.momentum_10 * 3.0)
    momentum_quality = _clamp(rsi_score * 0.5 + mom_score * 0.5)

    # 3. volume_flow: vol_ratio → 0-100
    # vol_ratio of 1.0 = average → 50; 2.0 = strong → 80
    volume_flow = _clamp(ind.vol_ratio * 40.0)

    # 4. volatility_regime: atr_ratio + bb_width_pct
    # Low ATR ratio = compression = opportunity; moderate BB width is ideal
    vol_from_atr = _clamp(100.0 - ind.atr_ratio * 50.0)  # Lower atr_ratio = higher score
    vol_from_bb = _clamp(50.0 + (5.0 - ind.bb_width_pct) * 10.0)  # Moderate BB = high
    volatility_regime = _clamp(vol_from_atr * 0.5 + vol_from_bb * 0.5)

    # 5. risk_reward: distance to support/resistance + ATR
    # Close near support in uptrend = good R:R; near resistance in downtrend = good R:R
    range_20 = ind.high_20 - ind.low_20
    if range_20 > 0:
        dist_from_low = (ind.close - ind.low_20) / range_20
        dist_from_high = (ind.high_20 - ind.close) / range_20
        if regime in (MarketRegime.UPTREND, MarketRegime.STRONG_UPTREND):
            # Bullish: want close near support (low dist_from_low)
            rr_score = _clamp(80.0 - dist_from_low * 60.0)
        elif regime in (MarketRegime.DOWNTREND, MarketRegime.STRONG_DOWNTREND):
            # Bearish: want close near resistance (low dist_from_high)
            rr_score = _clamp(80.0 - dist_from_high * 60.0)
        else:
            rr_score = 50.0
    else:
        rr_score = 50.0
    risk_reward = rr_score

    # 6. catalyst_awareness: constant 50 (no catalyst data in backtest)
    catalyst_awareness = 50.0

    # 7. liquidity_execution: volume level → 0-100
    # Use vol_ratio as proxy (higher volume = better liquidity)
    liquidity_execution = _clamp(ind.vol_ratio * 35.0 + 20.0)

    # 8. sentiment_positioning: close_in_range + RSI
    # close_in_range near 1.0 = bullish sentiment; near 0 = bearish
    sent_from_range = ind.close_in_range * 100.0
    sent_from_rsi = ind.rsi14  # RSI directly as sentiment proxy
    sentiment_positioning = _clamp(sent_from_range * 0.5 + sent_from_rsi * 0.5)

    return {
        ScoreCategory.TREND_STRUCTURE: round(trend_structure, 2),
        ScoreCategory.MOMENTUM_QUALITY: round(momentum_quality, 2),
        ScoreCategory.VOLUME_FLOW: round(volume_flow, 2),
        ScoreCategory.VOLATILITY_REGIME: round(volatility_regime, 2),
        ScoreCategory.RISK_REWARD: round(risk_reward, 2),
        ScoreCategory.CATALYST_AWARENESS: round(catalyst_awareness, 2),
        ScoreCategory.LIQUIDITY_EXECUTION: round(liquidity_execution, 2),
        ScoreCategory.SENTIMENT_POSITIONING: round(sentiment_positioning, 2),
    }


# ── Model Configuration ────────────────────────────────────────────


@dataclass(frozen=True)
class StopConfig:
    """Stop-loss and exit parameters for a model."""

    hard_stop_pct: float = -0.35
    trailing_stop_pct: float = 0.35
    take_profit_pct: float = 2.50
    max_hold_days: int = 9


@dataclass
class TournamentModel:
    """Base tournament model with scoring weights and rules."""

    name: ModelName
    weights: dict[str, float]
    threshold: float
    stop_config: StopConfig = field(default_factory=StopConfig)
    # Optional hard gates (indicator name → min value)
    hard_gates: dict[str, float] = field(default_factory=dict)
    # Optional features
    invert_rsi_at_extremes: bool = False
    skip_adx_above: float | None = None
    ema9_exit: bool = False
    no_followthrough_days: int | None = None

    def score_setup(
        self,
        ind: Indicators,
        regime: MarketRegime,
        category_scores: dict[str, float] | None = None,
    ) -> float:
        """Score a setup from 0-100 using weighted category scores.

        Args:
            ind: Technical indicators for the current bar.
            regime: Detected market regime.
            category_scores: Pre-computed category scores (optional, computed if None).

        Returns:
            Weighted composite score from 0 to 100.
        """
        if category_scores is None:
            category_scores = compute_category_scores(ind, regime)

        scores = dict(category_scores)

        # Apply RSI inversion at extremes for contrarian models
        if self.invert_rsi_at_extremes:
            if ind.rsi14 < 25:
                # Oversold = bullish for contrarian → boost momentum + sentiment
                scores[ScoreCategory.MOMENTUM_QUALITY] = _clamp(
                    100.0 - scores[ScoreCategory.MOMENTUM_QUALITY] * 0.5
                )
                scores[ScoreCategory.SENTIMENT_POSITIONING] = _clamp(
                    100.0 - scores[ScoreCategory.SENTIMENT_POSITIONING] * 0.5
                )
            elif ind.rsi14 > 75:
                # Overbought = bearish for contrarian → boost inverted
                scores[ScoreCategory.MOMENTUM_QUALITY] = _clamp(
                    100.0 - scores[ScoreCategory.MOMENTUM_QUALITY] * 0.5
                )
                scores[ScoreCategory.SENTIMENT_POSITIONING] = _clamp(
                    100.0 - scores[ScoreCategory.SENTIMENT_POSITIONING] * 0.5
                )

        total = 0.0
        weight_sum = 0.0
        for cat, weight in self.weights.items():
            score = scores.get(cat, 50.0)
            total += score * weight
            weight_sum += weight

        if weight_sum > 0:
            return round(total / weight_sum, 2)
        return 0.0

    def should_enter(self, score: float, ind: Indicators) -> bool:
        """Check if the model wants to enter this trade.

        Applies threshold check plus any hard gates.
        """
        if score < self.threshold:
            return False

        # Hard gates
        for gate_key, gate_min in self.hard_gates.items():
            val = getattr(ind, gate_key, None)
            if val is not None and val < gate_min:
                return False

        # Skip entry if ADX is too high (contrarian models)
        return not (self.skip_adx_above is not None and ind.adx14 > self.skip_adx_above)


# ── Model Definitions ──────────────────────────────────────────────


def _make_weights(
    trend: float,
    momentum: float,
    volume: float,
    volatility: float,
    risk_reward: float,
    catalyst: float,
    liquidity: float,
    sentiment: float,
) -> dict[str, float]:
    """Create a weight dict from shorthand values."""
    return {
        ScoreCategory.TREND_STRUCTURE: trend,
        ScoreCategory.MOMENTUM_QUALITY: momentum,
        ScoreCategory.VOLUME_FLOW: volume,
        ScoreCategory.VOLATILITY_REGIME: volatility,
        ScoreCategory.RISK_REWARD: risk_reward,
        ScoreCategory.CATALYST_AWARENESS: catalyst,
        ScoreCategory.LIQUIDITY_EXECUTION: liquidity,
        ScoreCategory.SENTIMENT_POSITIONING: sentiment,
    }


def build_edge_core() -> TournamentModel:
    """EDGE_CORE — Balanced generalist model."""
    return TournamentModel(
        name=ModelName.EDGE_CORE,
        weights=_make_weights(15, 15, 15, 10, 15, 10, 10, 10),
        threshold=72.0,
        stop_config=StopConfig(
            hard_stop_pct=-0.35,
            trailing_stop_pct=0.35,
            take_profit_pct=2.50,
            max_hold_days=9,
        ),
    )


def build_momentum_alpha() -> TournamentModel:
    """MOMENTUM_ALPHA — Aggressive trend follower."""
    return TournamentModel(
        name=ModelName.MOMENTUM_ALPHA,
        weights=_make_weights(30, 25, 10, 5, 15, 5, 5, 5),
        threshold=68.0,
        hard_gates={"adx14": 25.0},
        ema9_exit=True,
        stop_config=StopConfig(
            hard_stop_pct=-0.40,
            trailing_stop_pct=0.50,  # 2x ATR trailing
            take_profit_pct=3.50,
            max_hold_days=21,
        ),
    )


def build_flow_hunter() -> TournamentModel:
    """FLOW_HUNTER — Institutional flow follower."""
    return TournamentModel(
        name=ModelName.FLOW_HUNTER,
        weights=_make_weights(5, 5, 40, 10, 10, 10, 15, 5),
        threshold=70.0,
        no_followthrough_days=3,
        stop_config=StopConfig(
            hard_stop_pct=-0.35,
            trailing_stop_pct=0.30,
            take_profit_pct=2.00,
            max_hold_days=7,
        ),
    )


def build_contrarian_edge() -> TournamentModel:
    """CONTRARIAN_EDGE — Mean reversion contrarian."""
    return TournamentModel(
        name=ModelName.CONTRARIAN_EDGE,
        weights=_make_weights(5, 10, 10, 20, 15, 10, 10, 20),
        threshold=75.0,
        invert_rsi_at_extremes=True,
        skip_adx_above=30.0,
        stop_config=StopConfig(
            hard_stop_pct=-0.60,  # Wider stops
            trailing_stop_pct=0.50,  # Lower TP
            take_profit_pct=1.50,
            max_hold_days=7,
        ),
    )


def build_regime_chameleon(
    regime: MarketRegime,
    all_models: list[TournamentModel] | None = None,
) -> TournamentModel:
    """REGIME_CHAMELEON — Regime-adaptive morphing model.

    Detects the current regime and adopts the weights of the model
    best suited for that regime:
    - Strong trend → MOMENTUM_ALPHA weights
    - Sideways → CONTRARIAN_EDGE weights
    - Moderate trend → EDGE_CORE weights
    - High volume regime → FLOW_HUNTER weights
    """
    if all_models is None:
        all_models = [
            build_edge_core(),
            build_momentum_alpha(),
            build_flow_hunter(),
            build_contrarian_edge(),
        ]

    model_map: dict[str, TournamentModel] = {m.name.value: m for m in all_models}

    # Pick weights based on regime
    if regime in (MarketRegime.STRONG_UPTREND, MarketRegime.STRONG_DOWNTREND):
        donor = model_map.get(ModelName.MOMENTUM_ALPHA.value, all_models[0])
    elif regime == MarketRegime.SIDEWAYS:
        donor = model_map.get(ModelName.CONTRARIAN_EDGE.value, all_models[0])
    elif regime in (MarketRegime.UPTREND, MarketRegime.DOWNTREND):
        donor = model_map.get(ModelName.EDGE_CORE.value, all_models[0])
    else:
        donor = model_map.get(ModelName.EDGE_CORE.value, all_models[0])

    return TournamentModel(
        name=ModelName.REGIME_CHAMELEON,
        weights=dict(donor.weights),
        threshold=70.0,
        stop_config=donor.stop_config,
        hard_gates=dict(donor.hard_gates),
        invert_rsi_at_extremes=donor.invert_rsi_at_extremes,
        skip_adx_above=donor.skip_adx_above,
        ema9_exit=donor.ema9_exit,
        no_followthrough_days=donor.no_followthrough_days,
    )


def build_all_models(regime: MarketRegime | None = None) -> list[TournamentModel]:
    """Build all 5 tournament models.

    The REGIME_CHAMELEON requires a regime to select its donor model.
    If regime is None, defaults to EDGE_CORE weights.
    """
    core = build_edge_core()
    momentum = build_momentum_alpha()
    flow = build_flow_hunter()
    contrarian = build_contrarian_edge()
    effective_regime = regime if regime is not None else MarketRegime.SIDEWAYS
    chameleon = build_regime_chameleon(
        effective_regime,
        [core, momentum, flow, contrarian],
    )
    return [core, momentum, flow, contrarian, chameleon]
