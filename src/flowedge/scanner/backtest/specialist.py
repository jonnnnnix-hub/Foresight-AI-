"""Ticker specialist bots — per-ticker optimized trading bots.

Each specialist is an expert on 1-3 tickers with:
- Optimized entry/exit parameters (from grid search)
- Instrument preference (shares vs options)
- Historical performance baseline
- Custom conviction scoring

Specialist configs are auto-generated from optimizer results or manually tuned.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import structlog

from flowedge.scanner.backtest.schemas import BacktestResult

logger = structlog.get_logger()

SPECIALISTS_DIR = Path("data/specialists")


@dataclass
class SpecialistConfig:
    """Configuration for a ticker specialist bot."""

    name: str
    tickers: list[str]
    instrument: str  # "shares", "options", "both"
    shares_params: dict[str, float] = field(default_factory=dict)
    scalp_params: dict[str, float] = field(default_factory=dict)
    baseline_win_rate: float = 0.0
    optimized_win_rate: float = 0.0
    baseline_return_pct: float = 0.0
    optimized_return_pct: float = 0.0
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "tickers": self.tickers,
            "instrument": self.instrument,
            "shares_params": self.shares_params,
            "scalp_params": self.scalp_params,
            "baseline_win_rate": self.baseline_win_rate,
            "optimized_win_rate": self.optimized_win_rate,
            "baseline_return_pct": self.baseline_return_pct,
            "optimized_return_pct": self.optimized_return_pct,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SpecialistConfig:
        return cls(
            name=data["name"],
            tickers=data["tickers"],
            instrument=data.get("instrument", "shares"),
            shares_params=data.get("shares_params", {}),
            scalp_params=data.get("scalp_params", {}),
            baseline_win_rate=data.get("baseline_win_rate", 0.0),
            optimized_win_rate=data.get("optimized_win_rate", 0.0),
            baseline_return_pct=data.get("baseline_return_pct", 0.0),
            optimized_return_pct=data.get("optimized_return_pct", 0.0),
            notes=data.get("notes", ""),
        )


@dataclass
class SpecialistResult:
    """Result of running a specialist bot backtest."""

    config: SpecialistConfig
    shares_result: BacktestResult | None = None
    scalp_result: BacktestResult | None = None

    @property
    def combined_win_rate(self) -> float:
        trades = 0
        wins = 0
        if self.shares_result and self.shares_result.total_trades > 0:
            trades += self.shares_result.total_trades
            wins += self.shares_result.wins
        if self.scalp_result and self.scalp_result.total_trades > 0:
            trades += self.scalp_result.total_trades
            wins += self.scalp_result.wins
        return wins / trades if trades > 0 else 0.0

    @property
    def combined_return_pct(self) -> float:
        ret = 0.0
        if self.shares_result:
            ret += self.shares_result.portfolio_return_pct
        if self.scalp_result:
            ret += self.scalp_result.portfolio_return_pct
        return ret

    @property
    def total_trades(self) -> int:
        t = 0
        if self.shares_result:
            t += self.shares_result.total_trades
        if self.scalp_result:
            t += self.scalp_result.total_trades
        return t


# ── Config Generation ────────────────────────────────────────────────────────


def generate_specialists_from_optimizer(
    shares_results_path: Path | None = None,
    scalp_results_path: Path | None = None,
    min_win_rate: float = 0.55,
    min_trades: int = 10,
) -> list[SpecialistConfig]:
    """Auto-generate specialist configs from grid search results.

    Creates a specialist for each ticker that meets the minimum thresholds.
    """
    shares_path = shares_results_path or Path("data/optimizer/shares_grid_results.json")
    scalp_path = scalp_results_path or Path("data/optimizer/scalp_grid_results.json")

    shares_data: dict[str, Any] = {}
    scalp_data: dict[str, Any] = {}

    if shares_path.exists():
        shares_data = json.loads(shares_path.read_text())
    if scalp_path.exists():
        scalp_data = json.loads(scalp_path.read_text())

    # Index by ticker
    shares_by_ticker: dict[str, dict[str, Any]] = {}
    for t in shares_data.get("tickers", []):
        shares_by_ticker[t["ticker"]] = t

    scalp_by_ticker: dict[str, dict[str, Any]] = {}
    for t in scalp_data.get("tickers", []):
        scalp_by_ticker[t["ticker"]] = t

    all_tickers = set(shares_by_ticker.keys()) | set(scalp_by_ticker.keys())
    specialists: list[SpecialistConfig] = []

    for ticker in sorted(all_tickers):
        shares = shares_by_ticker.get(ticker)
        scalp = scalp_by_ticker.get(ticker)

        shares_viable = (
            shares is not None
            and shares.get("optimized_win_rate", 0) >= min_win_rate
            and shares.get("total_trades", 0) >= min_trades
        )
        scalp_viable = (
            scalp is not None
            and scalp.get("optimized_win_rate", 0) >= min_win_rate
            and scalp.get("total_trades", 0) >= max(5, min_trades // 2)
        )

        if not shares_viable and not scalp_viable:
            continue

        if shares_viable and scalp_viable:
            instrument = "both"
        elif shares_viable:
            instrument = "shares"
        else:
            instrument = "options"

        best_wr = 0.0
        best_ret = 0.0
        base_wr = 0.0
        base_ret = 0.0

        shares_params: dict[str, float] = {}
        scalp_params: dict[str, float] = {}

        if shares_viable and shares:
            shares_params = shares["best_params"]
            if shares["optimized_win_rate"] > best_wr:
                best_wr = shares["optimized_win_rate"]
                best_ret = shares["optimized_return_pct"]
                base_wr = shares["baseline_win_rate"]
                base_ret = shares["baseline_return_pct"]

        if scalp_viable and scalp:
            scalp_params = scalp["best_params"]
            if scalp["optimized_win_rate"] > best_wr:
                best_wr = scalp["optimized_win_rate"]
                best_ret = scalp["optimized_return_pct"]
                base_wr = scalp["baseline_win_rate"]
                base_ret = scalp["baseline_return_pct"]

        spec = SpecialistConfig(
            name=f"{ticker}_specialist",
            tickers=[ticker],
            instrument=instrument,
            shares_params=shares_params,
            scalp_params=scalp_params,
            baseline_win_rate=base_wr,
            optimized_win_rate=best_wr,
            baseline_return_pct=base_ret,
            optimized_return_pct=best_ret,
        )
        specialists.append(spec)

    logger.info(
        "specialists_generated",
        count=len(specialists),
        shares_only=sum(1 for s in specialists if s.instrument == "shares"),
        options_only=sum(1 for s in specialists if s.instrument == "options"),
        both=sum(1 for s in specialists if s.instrument == "both"),
    )

    return specialists


# ── Running Specialists ──────────────────────────────────────────────────────


def run_specialist(
    config: SpecialistConfig,
    starting_capital: float = 10_000.0,
) -> SpecialistResult:
    """Run a specialist bot backtest with its optimized parameters."""
    from flowedge.scanner.backtest.shares_engine import run_shares_backtest

    result = SpecialistResult(config=config)

    if config.instrument in ("shares", "both") and config.shares_params:
        result.shares_result = run_shares_backtest(
            mode="precision_shares",
            starting_capital=starting_capital,
            tickers=config.tickers,
            params=config.shares_params,
        )
        logger.info(
            "specialist_shares_done",
            name=config.name,
            trades=result.shares_result.total_trades,
            win_rate=result.shares_result.win_rate,
            return_pct=result.shares_result.portfolio_return_pct,
        )

    if config.instrument in ("options", "both") and config.scalp_params:
        from flowedge.scanner.backtest.scalp_real import run_scalp_real_backtest

        result.scalp_result = run_scalp_real_backtest(
            tickers=config.tickers,
            starting_capital=starting_capital,
            params=config.scalp_params,
        )
        logger.info(
            "specialist_scalp_done",
            name=config.name,
            trades=result.scalp_result.total_trades,
            win_rate=result.scalp_result.win_rate,
            return_pct=result.scalp_result.portfolio_return_pct,
        )

    return result


def run_all_specialists(
    specialists: list[SpecialistConfig] | None = None,
    starting_capital: float = 10_000.0,
) -> list[SpecialistResult]:
    """Run all specialist bots and report results."""
    if specialists is None:
        specialists = generate_specialists_from_optimizer()

    if not specialists:
        logger.warning("no_specialists", msg="No specialist configs found")
        return []

    results: list[SpecialistResult] = []
    for spec in specialists:
        result = run_specialist(spec, starting_capital=starting_capital)
        results.append(result)

    # Sort by combined WR
    results.sort(key=lambda r: r.combined_win_rate, reverse=True)
    _print_specialist_report(results)

    return results


# ── Persistence ──────────────────────────────────────────────────────────────


def save_specialists(specialists: list[SpecialistConfig]) -> None:
    """Save specialist configs to data/specialists/."""
    SPECIALISTS_DIR.mkdir(parents=True, exist_ok=True)

    for spec in specialists:
        path = SPECIALISTS_DIR / f"{spec.tickers[0]}_config.json"
        path.write_text(json.dumps(spec.to_dict(), indent=2))

    # Also save an index
    index = {
        "specialists": [
            {
                "name": s.name,
                "tickers": s.tickers,
                "instrument": s.instrument,
                "optimized_win_rate": s.optimized_win_rate,
            }
            for s in specialists
        ]
    }
    (SPECIALISTS_DIR / "index.json").write_text(json.dumps(index, indent=2))
    logger.info("specialists_saved", count=len(specialists), dir=str(SPECIALISTS_DIR))


def load_specialists() -> list[SpecialistConfig]:
    """Load specialist configs from data/specialists/."""
    if not SPECIALISTS_DIR.exists():
        return []

    configs: list[SpecialistConfig] = []
    for f in sorted(SPECIALISTS_DIR.glob("*_config.json")):
        data = json.loads(f.read_text())
        configs.append(SpecialistConfig.from_dict(data))

    return configs


# ── Reporting ────────────────────────────────────────────────────────────────


def _print_specialist_report(results: list[SpecialistResult]) -> None:
    """Print specialist bot backtest results."""
    print("\n" + "=" * 100)
    print(f"SPECIALIST BOTS — {len(results)} bots")
    print("=" * 100)
    print(
        f"{'Name':<20} {'Instrument':<10} {'WR':>7} {'Return':>9} "
        f"{'Trades':>7} {'Shares WR':>10} {'Scalp WR':>10}"
    )
    print("-" * 100)

    for r in results:
        shr_wr = f"{r.shares_result.win_rate:.1%}" if r.shares_result else "—"
        scl_wr = f"{r.scalp_result.win_rate:.1%}" if r.scalp_result else "—"
        print(
            f"{r.config.name:<20} "
            f"{r.config.instrument:<10} "
            f"{r.combined_win_rate:>6.1%} "
            f"{r.combined_return_pct:>+8.1f}% "
            f"{r.total_trades:>7} "
            f"{shr_wr:>10} "
            f"{scl_wr:>10}"
        )

    print("=" * 100 + "\n")


# ── CLI ──────────────────────────────────────────────────────────────────────


if __name__ == "__main__":
    import sys

    from flowedge.config.logging import setup_logging

    setup_logging("INFO")

    cmd = sys.argv[1] if len(sys.argv) > 1 else "run"

    if cmd == "generate":
        specs = generate_specialists_from_optimizer()
        save_specialists(specs)
        for s in specs:
            print(f"  {s.name}: {s.instrument}, WR {s.optimized_win_rate:.1%}")
    elif cmd == "run":
        specs = load_specialists()
        if not specs:
            specs = generate_specialists_from_optimizer()
            save_specialists(specs)
        run_all_specialists(specs)
    else:
        print("Usage: python -m flowedge.scanner.backtest.specialist [generate|run]")
        sys.exit(1)
