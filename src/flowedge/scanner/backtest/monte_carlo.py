"""Monte Carlo price path simulation for trade conviction scoring.

Generates N simulated future price paths using Geometric Brownian Motion
(GBM) calibrated from recent historical volatility and drift. For each
candidate trade, simulates 50,000 paths to estimate:

1. Probability of profit (P(profit)) — fraction of paths where the option
   would be profitable at the target hold period
2. Expected directional move — mean % change across all paths
3. Probability of stop-out — fraction of paths hitting hard stop
4. Confidence interval — 5th/95th percentile price range

These become conviction modifiers: high P(profit) boosts entry, low
P(profit) blocks entry.

No external dependencies — uses stdlib random and math only.
"""

from __future__ import annotations

import math
import random
from typing import Any


def _estimate_params(
    bars: list[dict[str, Any]],
    lookback: int = 30,
) -> tuple[float, float]:
    """Estimate annualized drift (mu) and volatility (sigma) from recent bars.

    Uses log returns over the lookback period.
    """
    closes = [float(b.get("close", 0)) for b in bars[-lookback:] if float(b.get("close", 0)) > 0]
    if len(closes) < 5:
        return 0.0, 0.30  # Default: no drift, 30% vol

    log_returns: list[float] = []
    for i in range(1, len(closes)):
        if closes[i - 1] > 0:
            log_returns.append(math.log(closes[i] / closes[i - 1]))

    if not log_returns:
        return 0.0, 0.30

    n = len(log_returns)
    mu_daily = sum(log_returns) / n
    var_daily = sum((r - mu_daily) ** 2 for r in log_returns) / max(n - 1, 1)
    sigma_daily = math.sqrt(var_daily) if var_daily > 0 else 0.01

    # Annualize
    mu_annual = mu_daily * 252
    sigma_annual = sigma_daily * math.sqrt(252)

    # Clamp to reasonable ranges
    mu_annual = max(-1.0, min(1.0, mu_annual))
    sigma_annual = max(0.05, min(2.0, sigma_annual))

    return mu_annual, sigma_annual


def simulate_paths(
    current_price: float,
    mu: float,
    sigma: float,
    days: int = 10,
    n_paths: int = 50_000,
    seed: int | None = None,
) -> list[list[float]]:
    """Generate N price paths using Geometric Brownian Motion.

    Each path is a list of daily closes for `days` trading days.

    GBM: S(t+dt) = S(t) * exp((mu - sigma^2/2)*dt + sigma*sqrt(dt)*Z)
    where Z ~ N(0,1)
    """
    if seed is not None:
        random.seed(seed)

    dt = 1.0 / 252  # One trading day
    drift = (mu - 0.5 * sigma * sigma) * dt
    diffusion = sigma * math.sqrt(dt)

    paths: list[list[float]] = []
    for _ in range(n_paths):
        path = [current_price]
        price = current_price
        for _ in range(days):
            z = random.gauss(0, 1)
            price = price * math.exp(drift + diffusion * z)
            path.append(price)
        paths.append(path)

    return paths


def analyze_paths(
    paths: list[list[float]],
    current_price: float,
    direction: str,
    otm_pct: float = 0.025,
    hold_days: int = 9,
    hard_stop_underlying_pct: float = 0.05,
) -> dict[str, float]:
    """Analyze simulated paths for trade viability.

    Returns dict with:
    - prob_profit: fraction of paths where trade would profit
    - prob_stop_out: fraction hitting hard stop on underlying
    - expected_move_pct: mean price change across paths
    - median_move_pct: median price change
    - confidence_5th: 5th percentile final price
    - confidence_95th: 95th percentile final price
    - max_adverse_pct: worst drawdown across paths (avg)
    """
    n = len(paths)
    if n == 0:
        return {
            "prob_profit": 0.0,
            "prob_stop_out": 1.0,
            "expected_move_pct": 0.0,
            "median_move_pct": 0.0,
            "confidence_5th": current_price,
            "confidence_95th": current_price,
            "max_adverse_pct": 0.0,
        }

    # For each path, determine if the option would profit
    # A call profits if price moves UP past strike; put profits if DOWN
    is_call = direction == "bullish"
    strike = current_price * (1.0 + otm_pct) if is_call else current_price * (1.0 - otm_pct)

    profitable_count = 0
    stop_out_count = 0
    final_moves: list[float] = []
    adverse_moves: list[float] = []

    check_day = min(hold_days, len(paths[0]) - 1)

    for path in paths:
        final_price = path[check_day]
        move_pct = (final_price - current_price) / current_price * 100
        final_moves.append(move_pct)

        # Check if option would be profitable
        profitable = final_price > strike if is_call else final_price < strike

        if profitable:
            profitable_count += 1

        # Check stop-out (underlying moves against direction)
        max_adverse = 0.0
        for day_price in path[1 : check_day + 1]:
            if is_call:
                adverse = (current_price - day_price) / current_price
            else:
                adverse = (day_price - current_price) / current_price
            if adverse > max_adverse:
                max_adverse = adverse

        adverse_moves.append(max_adverse)
        if max_adverse > hard_stop_underlying_pct:
            stop_out_count += 1

    final_moves.sort()
    median_idx = n // 2
    pct_5 = final_moves[int(n * 0.05)]
    pct_95 = final_moves[int(n * 0.95)]

    avg_adverse = sum(adverse_moves) / n if adverse_moves else 0.0

    return {
        "prob_profit": round(profitable_count / n, 4),
        "prob_stop_out": round(stop_out_count / n, 4),
        "expected_move_pct": round(sum(final_moves) / n, 4),
        "median_move_pct": round(final_moves[median_idx], 4),
        "confidence_5th": round(
            current_price * (1 + pct_5 / 100), 2
        ),
        "confidence_95th": round(
            current_price * (1 + pct_95 / 100), 2
        ),
        "max_adverse_pct": round(avg_adverse * 100, 4),
    }


def compute_mc_conviction(
    bars: list[dict[str, Any]],
    direction: str,
    otm_pct: float = 0.025,
    hold_days: int = 9,
    n_simulations: int = 50_000,
    seed: int | None = None,
) -> tuple[float, dict[str, float]]:
    """Run Monte Carlo simulation and return conviction adjustment.

    Returns:
        (adjustment, analysis_dict)
        adjustment: -3.0 to +3.0 conviction modifier
        analysis_dict: full MC analysis results
    """
    if len(bars) < 10:
        return 0.0, {"prob_profit": 0.5, "n_simulations": 0}

    current_price = float(bars[-1].get("close", 0))
    if current_price <= 0:
        return 0.0, {"prob_profit": 0.5, "n_simulations": 0}

    # Estimate parameters from historical data
    mu, sigma = _estimate_params(bars, lookback=30)

    # Run simulations
    paths = simulate_paths(
        current_price=current_price,
        mu=mu,
        sigma=sigma,
        days=hold_days,
        n_paths=n_simulations,
        seed=seed,
    )

    # Analyze results
    analysis = analyze_paths(
        paths=paths,
        current_price=current_price,
        direction=direction,
        otm_pct=otm_pct,
        hold_days=hold_days,
    )
    analysis["n_simulations"] = float(n_simulations)
    analysis["mu_annual"] = round(mu, 4)
    analysis["sigma_annual"] = round(sigma, 4)

    # Convert probability to conviction adjustment
    # NOTE: OTM options typically have P(profit) ~25-45% under GBM,
    # so we use a softer scale to avoid over-filtering.
    prob = analysis["prob_profit"]
    prob_stop = analysis["prob_stop_out"]

    # Soft adjustment: ±1.5 range to complement other factors
    if prob >= 0.55:
        adjustment = 1.5  # Strong statistical edge — boost
    elif prob >= 0.45:
        adjustment = 0.75  # Favorable odds
    elif prob >= 0.35:
        adjustment = 0.0  # Neutral — typical for OTM options
    elif prob >= 0.25:
        adjustment = -0.5  # Below average — mild penalty
    elif prob >= 0.15:
        adjustment = -1.0  # Poor odds
    else:
        adjustment = -1.5  # Very low — strong penalty

    # Mild additional penalty for extreme stop-out risk
    if prob_stop > 0.60:
        adjustment -= 0.5
    elif prob_stop > 0.50:
        adjustment -= 0.25

    return round(max(-1.5, min(1.5, adjustment)), 1), analysis
