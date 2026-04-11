"""Scanner CLI commands — scan for options opportunities."""

from __future__ import annotations

import asyncio
import json
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from flowedge.config.logging import setup_logging
from flowedge.config.settings import get_settings
from flowedge.scanner.catalyst.engine import scan_catalysts
from flowedge.scanner.iv_rank.engine import scan_iv
from flowedge.scanner.providers.registry import ProviderRegistry
from flowedge.scanner.schemas.signals import ScannerResult
from flowedge.scanner.scorer.engine import score_lottos
from flowedge.scanner.uoa.engine import scan_uoa

scanner_app = typer.Typer(name="scan", help="Options scanner commands")
console = Console()


@scanner_app.command("run")
def run_scan(
    tickers: Annotated[list[str], typer.Argument(help="Tickers to scan")],
    min_score: Annotated[float, typer.Option(help="Min composite score")] = 0.0,
    output_format: Annotated[str, typer.Option(help="table | json")] = "table",
    log_level: Annotated[str, typer.Option(help="Log level")] = "INFO",
) -> None:
    """Scan tickers for lotto play opportunities."""
    setup_logging(log_level)
    result = asyncio.run(_run_full_scan([t.upper() for t in tickers]))

    filtered = [
        o for o in result.top_opportunities if o.composite_score >= min_score
    ]

    if output_format == "json":
        console.print(json.dumps(
            [o.model_dump(mode="json") for o in filtered],
            indent=2,
            default=str,
        ))
        return

    if not filtered:
        console.print("[yellow]No opportunities above minimum score.[/yellow]")
        return

    table = Table(title=f"Lotto Opportunities (min score: {min_score})")
    table.add_column("Ticker", style="bold")
    table.add_column("Score", justify="right")
    table.add_column("UOA", justify="right")
    table.add_column("IV", justify="right")
    table.add_column("Cat", justify="right")
    table.add_column("Direction")
    table.add_column("Rationale")

    for opp in filtered:
        direction_color = {
            "bullish": "green",
            "bearish": "red",
            "neutral": "yellow",
        }.get(opp.suggested_direction.value, "white")

        table.add_row(
            opp.ticker,
            f"{opp.composite_score:.1f}",
            f"{opp.uoa_score:.1f}",
            f"{opp.iv_score:.1f}",
            f"{opp.catalyst_score:.1f}",
            f"[{direction_color}]{opp.suggested_direction.value}[/{direction_color}]",
            opp.rationale[:80] if opp.rationale else "",
        )

    console.print(table)
    console.print(f"\n[green]{len(filtered)} opportunities found[/green]")

    if result.errors:
        console.print("\n[yellow]Warnings:[/yellow]")
        for err in result.errors:
            console.print(f"  [red]- {err}[/red]")


@scanner_app.command("flow")
def flow_cmd(
    ticker: Annotated[str, typer.Argument(help="Ticker")],
    log_level: Annotated[str, typer.Option(help="Log level")] = "WARNING",
) -> None:
    """Show unusual options flow for a ticker."""
    setup_logging(log_level)
    settings = get_settings()
    registry = ProviderRegistry(settings)

    signals = asyncio.run(scan_uoa(registry, [ticker.upper()], settings))

    if not signals:
        console.print(f"[yellow]No unusual flow for {ticker.upper()}[/yellow]")
        return

    for sig in signals:
        console.print(f"\n[bold]{sig.ticker}[/bold] — {sig.signal_type}")
        console.print(f"  Direction: {sig.direction.value}")
        console.print(f"  Strength: {sig.strength:.1f}/10")
        console.print(f"  Total premium: ${sig.total_premium:,.0f}")
        console.print(f"  C/P ratio: {sig.call_put_ratio:.2f}")
        console.print(f"  Alerts: {len(sig.alerts)}")


@scanner_app.command("iv")
def iv_cmd(
    ticker: Annotated[str, typer.Argument(help="Ticker")],
    log_level: Annotated[str, typer.Option(help="Log level")] = "WARNING",
) -> None:
    """Show IV rank and regime for a ticker."""
    setup_logging(log_level)
    settings = get_settings()
    registry = ProviderRegistry(settings)

    signals = asyncio.run(scan_iv(registry, [ticker.upper()], settings))

    if not signals:
        console.print(f"[yellow]No IV data for {ticker.upper()}[/yellow]")
        return

    sig = signals[0]
    console.print(f"\n[bold]{sig.ticker}[/bold] — IV Analysis")
    console.print(f"  IV Rank: {sig.iv_rank.iv_rank:.1f}%")
    console.print(f"  IV Percentile: {sig.iv_rank.iv_percentile:.1f}%")
    console.print(f"  Current IV: {sig.iv_rank.current_iv:.4f}")
    console.print(f"  Regime: {sig.regime.value}")
    console.print(f"  Cheap premium: {'Yes' if sig.is_cheap_premium else 'No'}")
    console.print(f"  Term structure: {'Contango' if sig.is_contango else 'Backwardation'}")
    console.print(f"  Score: {sig.strength:.1f}/10")


@scanner_app.command("catalysts")
def catalysts_cmd(
    tickers: Annotated[list[str], typer.Argument(help="Tickers to check")],
    days_ahead: Annotated[int, typer.Option(help="Days to look ahead")] = 14,
    log_level: Annotated[str, typer.Option(help="Log level")] = "WARNING",
) -> None:
    """Show upcoming catalysts for tickers."""
    setup_logging(log_level)
    settings = get_settings()
    settings.catalyst_lookforward_days = days_ahead
    registry = ProviderRegistry(settings)

    signals = asyncio.run(
        scan_catalysts(registry, [t.upper() for t in tickers], settings)
    )

    if not signals:
        console.print("[yellow]No catalysts found.[/yellow]")
        return

    for sig in signals:
        console.print(f"\n[bold]{sig.ticker}[/bold]")
        if sig.earnings:
            for e in sig.earnings:
                console.print(f"  Earnings: {e.report_date} ({e.time_of_day})")
        if sig.expected_move:
            console.print(
                f"  Expected move: {sig.expected_move.expected_move_pct:.1f}%"
            )
        if sig.insider_trades:
            console.print(
                f"  Insiders: {sig.insider_buy_count}B / "
                f"{sig.insider_sell_count}S "
                f"(net ${sig.insider_net_value:+,.0f})"
            )
        console.print(f"  Score: {sig.strength:.1f}/10")


async def _run_full_scan(tickers: list[str]) -> ScannerResult:
    """Run all three scanners and composite scorer."""

    settings = get_settings()
    registry = ProviderRegistry(settings)

    try:
        uoa_signals = await scan_uoa(registry, tickers, settings)
        iv_signals = await scan_iv(registry, tickers, settings)
        catalyst_signals = await scan_catalysts(registry, tickers, settings)
        return score_lottos(uoa_signals, iv_signals, catalyst_signals, settings)
    finally:
        await registry.close_all()
