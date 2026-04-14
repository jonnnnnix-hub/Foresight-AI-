"""Black-Scholes option pricing for backtesting.

Uses ATR as IV proxy for historical data where implied volatility
is not directly available. Provides realistic option premium estimates
without requiring live options chain data.

No external dependencies — uses only stdlib math.
"""

from __future__ import annotations

from math import erf, exp, log, pi, sqrt


def norm_cdf(x: float) -> float:
    """Standard normal cumulative distribution function."""
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def norm_pdf(x: float) -> float:
    """Standard normal probability density function."""
    return exp(-0.5 * x * x) / sqrt(2.0 * pi)


def bs_price(
    s: float,
    k: float,
    t: float,
    r: float,
    sigma: float,
    is_call: bool = True,
) -> float:
    """Black-Scholes option price.

    Args:
        s: Current underlying price.
        k: Strike price.
        t: Time to expiration in years (e.g., 10/252 for 10 trading days).
        r: Risk-free rate (annualized, e.g., 0.05 for 5%).
        sigma: Implied volatility (annualized, e.g., 0.30 for 30%).
        is_call: True for call, False for put.

    Returns:
        Theoretical option price per share.
    """
    if t <= 0 or sigma <= 0 or s <= 0 or k <= 0:
        return max(0.0, s - k) if is_call else max(0.0, k - s)

    sqrt_t = sqrt(t)
    d1 = (log(s / k) + (r + sigma * sigma / 2.0) * t) / (sigma * sqrt_t)
    d2 = d1 - sigma * sqrt_t

    if is_call:
        return s * norm_cdf(d1) - k * exp(-r * t) * norm_cdf(d2)
    return k * exp(-r * t) * norm_cdf(-d2) - s * norm_cdf(-d1)


def bs_delta(
    s: float,
    k: float,
    t: float,
    r: float,
    sigma: float,
    is_call: bool = True,
) -> float:
    """Black-Scholes delta (∂C/∂S)."""
    if t <= 0 or sigma <= 0 or s <= 0 or k <= 0:
        return (1.0 if s > k else 0.0) if is_call else (-1.0 if s < k else 0.0)

    d1 = (log(s / k) + (r + sigma * sigma / 2.0) * t) / (sigma * sqrt(t))
    if is_call:
        return norm_cdf(d1)
    return norm_cdf(d1) - 1.0


def bs_gamma(
    s: float,
    k: float,
    t: float,
    r: float,
    sigma: float,
) -> float:
    """Black-Scholes gamma (∂²C/∂S²). Same for calls and puts."""
    if t <= 0 or sigma <= 0 or s <= 0 or k <= 0:
        return 0.0

    sqrt_t = sqrt(t)
    d1 = (log(s / k) + (r + sigma * sigma / 2.0) * t) / (sigma * sqrt_t)
    return norm_pdf(d1) / (s * sigma * sqrt_t)


def bs_theta(
    s: float,
    k: float,
    t: float,
    r: float,
    sigma: float,
    is_call: bool = True,
) -> float:
    """Black-Scholes theta (∂C/∂t) per calendar day."""
    if t <= 0 or sigma <= 0 or s <= 0 or k <= 0:
        return 0.0

    sqrt_t = sqrt(t)
    d1 = (log(s / k) + (r + sigma * sigma / 2.0) * t) / (sigma * sqrt_t)
    d2 = d1 - sigma * sqrt_t

    term1 = -(s * norm_pdf(d1) * sigma) / (2.0 * sqrt_t)
    term2 = (
        -r * k * exp(-r * t) * norm_cdf(d2)
        if is_call
        else r * k * exp(-r * t) * norm_cdf(-d2)
    )

    # Convert from per-year to per-calendar-day
    return (term1 + term2) / 365.0


def estimate_iv_from_atr(
    atr_value: float,
    close: float,
    period: int = 14,
) -> float:
    """Estimate annualized implied volatility from ATR.

    ATR/Close approximates daily realized volatility.
    IV typically trades at a premium to realized vol, so we
    scale by 1.25 to approximate the IV surface.

    Clamped to [0.10, 1.50] to avoid degenerate pricing.
    """
    if close <= 0 or atr_value <= 0:
        return 0.30  # Default 30%

    daily_vol = atr_value / close
    annualized = daily_vol * sqrt(252.0) * 1.25
    return max(0.10, min(1.50, annualized))


def estimate_premium(
    underlying: float,
    otm_pct: float,
    dte: int,
    iv: float,
    is_call: bool = True,
    r: float = 0.05,
) -> float:
    """Convenience: estimate option premium for an OTM option.

    Args:
        underlying: Current stock price.
        otm_pct: How far out of the money (e.g., 0.03 = 3%).
        dte: Days to expiration (trading days).
        iv: Annualized implied volatility.
        is_call: Call or put.
        r: Risk-free rate.

    Returns:
        Estimated premium per share.
    """
    strike = underlying * (1.0 + otm_pct) if is_call else underlying * (1.0 - otm_pct)

    t_years = max(dte, 1) / 252.0
    return bs_price(underlying, strike, t_years, r, iv, is_call)
