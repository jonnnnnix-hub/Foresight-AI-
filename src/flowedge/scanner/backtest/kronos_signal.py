"""Kronos-inspired price prediction signal for backtesting.

The full Kronos foundation model (NeoQuasar/Kronos-base) is a two-stage
transformer that predicts future candlestick bars. It requires:
- torch + HuggingFace transformers
- Pretrained tokenizer (NeoQuasar/Kronos-Tokenizer-base)
- Pretrained predictor (NeoQuasar/Kronos-small or Kronos-base)

For backtesting without GPU/torch, we implement a lightweight statistical
analog that captures the same concept: use recent price patterns to predict
directional probability for the next N bars.

Prediction approach:
1. Pattern matching: Compare recent N-bar pattern to all historical
   windows, find closest matches by normalized distance.
2. Outcome measurement: Look at what happened after similar patterns.
3. Directional confidence: Ratio of bullish vs bearish outcomes.
4. Conviction modifier: Strong directional prediction boosts/penalizes
   the entry signal conviction.

When torch is available, this module can be upgraded to use the real
Kronos model via `predict_with_kronos()`.
"""

from __future__ import annotations

from typing import Any


def _normalize_bars(bars: list[dict[str, Any]]) -> list[tuple[float, float, float, float]]:
    """Normalize OHLC bars to percentage changes from first bar's close.

    Returns list of (open_pct, high_pct, low_pct, close_pct).
    """
    if not bars:
        return []

    base = float(bars[0].get("close", 1))
    if base <= 0:
        base = 1.0

    result: list[tuple[float, float, float, float]] = []
    for b in bars:
        o = (float(b.get("open", 0)) - base) / base
        h = (float(b.get("high", 0)) - base) / base
        lo = (float(b.get("low", 0)) - base) / base
        c = (float(b.get("close", 0)) - base) / base
        result.append((o, h, lo, c))

    return result


def _pattern_distance(
    p1: list[tuple[float, float, float, float]],
    p2: list[tuple[float, float, float, float]],
) -> float:
    """Euclidean distance between two normalized bar patterns."""
    if len(p1) != len(p2):
        return float("inf")

    dist = 0.0
    for (o1, h1, l1, c1), (o2, h2, l2, c2) in zip(p1, p2, strict=True):
        dist += (o1 - o2) ** 2 + (h1 - h2) ** 2 + (l1 - l2) ** 2 + (c1 - c2) ** 2

    return float(dist**0.5)


def predict_direction(
    bars: list[dict[str, Any]],
    pattern_len: int = 10,
    forward_len: int = 5,
    top_k: int = 5,
) -> tuple[str, float]:
    """Predict direction using historical pattern matching.

    Scans all historical windows of length `pattern_len`, finds the
    top_k most similar to the current pattern, then measures what
    happened in the next `forward_len` bars after each match.

    Returns:
        (direction, confidence) where direction is "bullish"/"bearish"/"neutral"
        and confidence is 0.0-1.0.
    """
    min_history = pattern_len + forward_len + 20
    if len(bars) < min_history:
        return "neutral", 0.0

    # Current pattern = last `pattern_len` bars
    current_norm = _normalize_bars(bars[-pattern_len:])

    # Search all historical windows
    distances: list[tuple[float, int]] = []
    search_end = len(bars) - pattern_len - forward_len

    for i in range(0, search_end):
        window = bars[i : i + pattern_len]
        window_norm = _normalize_bars(window)
        dist = _pattern_distance(current_norm, window_norm)
        distances.append((dist, i))

    if not distances:
        return "neutral", 0.0

    # Sort by distance, take top_k matches
    distances.sort(key=lambda x: x[0])
    top_matches = distances[:top_k]

    # Measure outcomes after each match
    bullish_outcomes = 0
    bearish_outcomes = 0
    total_move = 0.0

    for _, idx in top_matches:
        entry_close = float(bars[idx + pattern_len - 1].get("close", 0))
        exit_close = float(
            bars[min(idx + pattern_len + forward_len - 1, len(bars) - 1)].get("close", 0)
        )
        if entry_close <= 0:
            continue

        move_pct = (exit_close - entry_close) / entry_close * 100
        total_move += move_pct

        if move_pct > 0.5:
            bullish_outcomes += 1
        elif move_pct < -0.5:
            bearish_outcomes += 1

    total_valid = bullish_outcomes + bearish_outcomes
    if total_valid == 0:
        return "neutral", 0.0

    # Direction = majority vote
    if bullish_outcomes > bearish_outcomes:
        direction = "bullish"
        confidence = bullish_outcomes / max(total_valid, 1)
    elif bearish_outcomes > bullish_outcomes:
        direction = "bearish"
        confidence = bearish_outcomes / max(total_valid, 1)
    else:
        direction = "neutral"
        confidence = 0.3

    # Scale confidence: unanimous = 1.0, slight edge = 0.5
    avg_move = total_move / top_k if top_k > 0 else 0.0
    move_magnitude = min(abs(avg_move) / 3.0, 1.0)  # Normalize to 0-1
    confidence = confidence * 0.6 + move_magnitude * 0.4

    return direction, round(min(confidence, 1.0), 3)


def compute_kronos_adjustment(
    bars: list[dict[str, Any]],
    trade_direction: str,
) -> float:
    """Compute conviction adjustment from Kronos-style prediction.

    Positive if prediction aligns with trade direction.
    Negative if prediction opposes.
    Range: -1.5 to +1.5.
    """
    pred_dir, confidence = predict_direction(bars)

    if pred_dir == "neutral" or confidence < 0.4:
        return 0.0

    aligned = pred_dir == trade_direction

    if aligned:
        if confidence >= 0.8:
            return 1.5
        if confidence >= 0.6:
            return 1.0
        return 0.5

    # Opposing prediction
    if confidence >= 0.8:
        return -1.5
    if confidence >= 0.6:
        return -1.0
    return -0.5
