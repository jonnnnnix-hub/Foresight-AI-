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

    table = Table(title=f"NEXUS — Lotto Opportunities (min score: {min_score})")
    table.add_column("Ticker", style="bold")
    table.add_column("NEXUS", justify="right")
    table.add_column("SPECTER", justify="right")
    table.add_column("ORACLE", justify="right")
    table.add_column("SENTINEL", justify="right")
    table.add_column("Direction")
    table.add_column("Signals")

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


@scanner_app.command("momentum")
def momentum_cmd(
    ticker: Annotated[str, typer.Argument(help="Ticker")],
    log_level: Annotated[str, typer.Option(help="Log level")] = "WARNING",
) -> None:
    """Analyze multi-timeframe technical momentum."""
    from flowedge.scanner.momentum.engine import analyze_momentum

    setup_logging(log_level)
    signal = asyncio.run(analyze_momentum(ticker.upper()))

    bias_colors = {
        "strong_bullish": "green",
        "bullish": "green",
        "neutral": "yellow",
        "bearish": "red",
        "strong_bearish": "red",
    }
    color = bias_colors.get(signal.bias.value, "white")
    console.print(f"\n[bold]{signal.ticker} — Momentum[/bold]")
    console.print(f"  Bias: [{color}]{signal.bias.value.upper()}[/{color}]")
    console.print(f"  Score: {signal.strength:.1f}/10")
    console.print(f"  Trend aligned: {'Yes' if signal.trend_alignment else 'No'}")
    if signal.rsi_oversold:
        console.print("  [red]RSI OVERSOLD[/red]")
    if signal.rsi_overbought:
        console.print("  [green]RSI OVERBOUGHT[/green]")
    if signal.macd_crossover:
        console.print("  [green]MACD bullish crossover[/green]")
    console.print(f"  {signal.rationale}")


@scanner_app.command("interpret")
def interpret_cmd(
    tickers: Annotated[list[str], typer.Argument(help="Tickers to interpret")],
    max_count: Annotated[int, typer.Option(help="Max opportunities to interpret")] = 3,
    log_level: Annotated[str, typer.Option(help="Log level")] = "WARNING",
) -> None:
    """AI-generate trade theses for top opportunities."""
    setup_logging(log_level)
    asyncio.run(_interpret_tickers([t.upper() for t in tickers], max_count))


async def _interpret_tickers(tickers: list[str], max_count: int) -> None:
    from flowedge.scanner.interpreter.engine import interpret_batch

    settings = get_settings()
    registry = ProviderRegistry(settings)

    try:
        uoa = await scan_uoa(registry, tickers, settings)
        iv = await scan_iv(registry, tickers, settings)
        cat = await scan_catalysts(registry, tickers, settings)
        result = score_lottos(uoa, iv, cat, settings)

        theses = await interpret_batch(
            result.top_opportunities, max_interpret=max_count, settings=settings
        )

        for thesis in theses:
            conv_color = {
                "high": "green",
                "medium": "yellow",
                "low": "red",
                "avoid": "red",
            }.get(thesis.conviction.value, "white")

            console.print(f"\n{'='*60}")
            console.print(
                f"[bold]{thesis.ticker}[/bold] — "
                f"[{conv_color}]{thesis.conviction.value.upper()} CONVICTION[/{conv_color}]"
            )
            console.print(f"\n  {thesis.thesis_summary}")
            console.print(f"\n  [bold]Smart Money:[/bold] {thesis.smart_money_read}")
            console.print(f"  [bold]Catalyst:[/bold] {thesis.catalyst_narrative}")
            console.print(f"  [bold]IV:[/bold] {thesis.iv_context}")
            if thesis.gex_context:
                console.print(f"  [bold]GEX:[/bold] {thesis.gex_context}")
            console.print(f"\n  [green]Entry:[/green] {thesis.ideal_entry}")
            console.print(f"  [green]Target:[/green] {thesis.target_exit}")
            console.print(f"  [red]Stop:[/red] {thesis.stop_logic}")
            console.print(f"  [blue]Size:[/blue] {thesis.position_sizing_note}")

            if thesis.key_risks:
                console.print("\n  [yellow]Risks:[/yellow]")
                for risk in thesis.key_risks:
                    console.print(f"    ! {risk}")
    finally:
        await registry.close_all()


@scanner_app.command("watch")
def watch_cmd(
    tickers: Annotated[list[str], typer.Argument(help="Tickers to watch")],
    interval: Annotated[int, typer.Option(help="Minutes between scans")] = 15,
    min_score: Annotated[float, typer.Option(help="Min score to alert")] = 5.0,
    webhook: Annotated[str, typer.Option(help="Slack/Discord webhook URL")] = "",
    log_file: Annotated[str, typer.Option(help="Alert log file path")] = "",
    max_cycles: Annotated[int, typer.Option(help="Max scan cycles (0=forever)")] = 0,
    log_level: Annotated[str, typer.Option(help="Log level")] = "INFO",
) -> None:
    """Watch tickers on a recurring interval and alert on opportunities."""
    from pathlib import Path

    from flowedge.scanner.scheduler.engine import (
        AlertChannel,
        ConsoleAlert,
        FileAlert,
        WebhookAlert,
        run_scanner_loop,
    )

    setup_logging(log_level)

    channels: list[AlertChannel] = [ConsoleAlert()]
    if webhook:
        channels.append(WebhookAlert(webhook))
    if log_file:
        channels.append(FileAlert(Path(log_file)))

    console.print(
        f"[green]Starting scanner watch:[/green] "
        f"{len(tickers)} tickers, every {interval}m, min score {min_score}"
    )
    console.print(f"  Tickers: {', '.join(t.upper() for t in tickers)}")
    if webhook:
        console.print(f"  Webhook: {webhook[:50]}...")
    if log_file:
        console.print(f"  Log file: {log_file}")
    console.print("[yellow]Press Ctrl+C to stop[/yellow]\n")

    asyncio.run(
        run_scanner_loop(
            tickers=[t.upper() for t in tickers],
            interval_minutes=interval,
            min_score=min_score,
            channels=channels,
            max_cycles=max_cycles if max_cycles > 0 else None,
        )
    )


@scanner_app.command("pick")
def pick_cmd(
    ticker: Annotated[str, typer.Argument(help="Ticker to pick contracts for")],
    direction: Annotated[str, typer.Option(help="bullish | bearish")] = "bullish",
    log_level: Annotated[str, typer.Option(help="Log level")] = "WARNING",
) -> None:
    """Pick optimal contracts for a lotto play on a ticker."""
    setup_logging(log_level)
    asyncio.run(_pick_contracts(ticker.upper(), direction))


async def _pick_contracts(ticker: str, direction: str) -> None:
    """Run contract selection for a ticker."""
    from flowedge.scanner.selector.engine import select_contracts

    settings = get_settings()
    registry = ProviderRegistry(settings)

    try:
        # Get signals
        iv_signals = await scan_iv(registry, [ticker], settings)
        catalyst_signals = await scan_catalysts(registry, [ticker], settings)

        # Build a dummy opportunity for the selector
        from flowedge.scanner.schemas.flow import FlowSentiment
        from flowedge.scanner.schemas.signals import LottoOpportunity

        opp = LottoOpportunity(
            ticker=ticker,
            suggested_direction=(
                FlowSentiment.BULLISH
                if direction == "bullish"
                else FlowSentiment.BEARISH
            ),
            iv_signal=iv_signals[0] if iv_signals else None,
            catalyst_signal=catalyst_signals[0] if catalyst_signals else None,
        )

        # Get live chain
        chain_provider = registry.get_options_chain_provider()
        chain = await chain_provider.get_options_chain(ticker)

        if not chain.contracts:
            console.print(f"[red]No options chain data for {ticker}[/red]")
            return

        recs = select_contracts(opp, chain)

        if not recs:
            console.print(f"[yellow]No suitable contracts found for {ticker}[/yellow]")
            return

        console.print(f"\n[bold]{ticker} — {direction.upper()} Contract Picks[/bold]")
        console.print(f"  Underlying: ${chain.underlying_price:.2f}\n")

        from rich.table import Table

        table = Table()
        table.add_column("Type")
        table.add_column("Strike")
        table.add_column("Expiry")
        table.add_column("Bid")
        table.add_column("Ask")
        table.add_column("Delta")
        table.add_column("Vol")
        table.add_column("OI")
        table.add_column("R/R")
        table.add_column("Max Loss")
        table.add_column("Breakeven")

        for r in recs:
            c = r.contract
            table.add_row(
                c.option_type.value.upper(),
                f"${c.strike:.2f}",
                str(c.expiration),
                f"${c.bid:.2f}",
                f"${c.ask:.2f}",
                f"{c.delta:.2f}" if c.delta else "n/a",
                str(c.volume),
                str(c.open_interest),
                f"{r.risk_reward:.1f}x",
                f"${r.max_loss:.0f}",
                f"{r.breakeven_pct:.1f}%",
            )

        console.print(table)

        for r in recs:
            console.print(f"  → {r.reason}")

    finally:
        await registry.close_all()


@scanner_app.command("backtest")
def backtest_cmd(
    tickers: Annotated[list[str], typer.Argument(help="Tickers to backtest")],
    lookback: Annotated[int, typer.Option(help="Lookback days")] = 90,
    hold_days: Annotated[int, typer.Option(help="Max hold days per trade")] = 10,
    take_profit: Annotated[float, typer.Option(help="Take profit %")] = 100.0,
    stop_loss: Annotated[float, typer.Option(help="Stop loss %")] = -80.0,
    log_level: Annotated[str, typer.Option(help="Log level")] = "WARNING",
) -> None:
    """Backtest lotto signals against historical data."""
    from flowedge.scanner.backtest.engine import run_backtest

    setup_logging(log_level)
    result = asyncio.run(
        run_backtest(
            tickers=[t.upper() for t in tickers],
            lookback_days=lookback,
            max_hold_days=hold_days,
            take_profit_pct=take_profit,
            stop_loss_pct=stop_loss,
        )
    )

    console.print(f"\n[bold]Backtest Results — {lookback}d lookback[/bold]")
    console.print(f"  Trades: {result.total_trades}")
    console.print(f"  Win rate: {result.win_rate:.1%}")
    console.print(f"  Avg win: {result.avg_win_pct:+.1f}%")
    console.print(f"  Avg loss: {result.avg_loss_pct:+.1f}%")
    console.print(f"  Best: {result.best_trade_pct:+.1f}%")
    console.print(f"  Worst: {result.worst_trade_pct:+.1f}%")
    console.print(f"  Profit factor: {result.profit_factor:.2f}")
    console.print(f"  Total P&L: {result.total_pnl_pct:+.1f}%")


@scanner_app.command("simulate")
def simulate_cmd(
    capital: Annotated[float, typer.Option(help="Starting capital")] = 1000.0,
    log_level: Annotated[str, typer.Option(help="Log level")] = "INFO",
) -> None:
    """Run $1000 bot simulation from April 1 and show results."""
    from flowedge.scanner.performance.simulator import run_historical_simulation

    setup_logging(log_level)
    console.print("[bold]PHANTOM — Running Historical Simulation[/bold]")
    console.print(f"  Starting: ${capital:,.2f} on April 1, 2026\n")

    report = asyncio.run(run_historical_simulation(starting_capital=capital))

    ret_color = "green" if report.total_return_pct >= 0 else "red"
    console.print(f"\n[bold]{'='*50}[/bold]")
    console.print("[bold]PHANTOM RESULTS[/bold]")
    console.print(f"  Period: {report.start_date} → {report.end_date}")
    console.print(
        f"  Starting: [bold]${report.starting_capital:,.2f}[/bold]"
    )
    console.print(
        f"  Ending:   [{ret_color}][bold]${report.ending_value:,.2f}[/bold]"
        f"[/{ret_color}]"
    )
    console.print(
        f"  Return:   [{ret_color}]{report.total_return_pct:+.1f}% "
        f"(${report.total_return_dollars:+,.2f})[/{ret_color}]"
    )
    console.print(f"  Trades: {report.total_trades} "
                  f"({report.wins}W / {report.losses}L / {report.open_trades} open)")
    console.print(f"  Win rate: {report.win_rate:.1%}")
    console.print(f"  Profit factor: {report.profit_factor:.2f}")
    console.print(f"  Max drawdown: {report.max_drawdown_pct:.1f}%")
    console.print(f"  Avg hold: {report.avg_hold_days:.1f} days")
    console.print("\n  Dashboard: http://localhost:8000/performance/")


@scanner_app.command("gex")
def gex_cmd(
    ticker: Annotated[str, typer.Argument(help="Ticker for GEX analysis")],
    log_level: Annotated[str, typer.Option(help="Log level")] = "WARNING",
) -> None:
    """Analyze gamma exposure and market structure for a ticker."""
    from flowedge.scanner.gex.engine import compute_gex_profile

    setup_logging(log_level)
    settings = get_settings()
    registry = ProviderRegistry(settings)

    profile = asyncio.run(compute_gex_profile(ticker.upper(), registry, settings))

    regime_color = {
        "positive": "green",
        "negative": "red",
        "neutral": "yellow",
    }.get(profile.regime.value, "white")

    console.print(f"\n[bold]{profile.ticker} — GEX Profile[/bold]")
    console.print(f"  Price: ${profile.underlying_price:.2f}")
    console.print(
        f"  Regime: [{regime_color}]{profile.regime.value.upper()}[/{regime_color}]"
    )
    console.print(f"  Total GEX: {profile.total_gex:,.0f}")
    if profile.gex_flip_price:
        console.print(f"  GEX Flip: ${profile.gex_flip_price:.2f}")
    if profile.max_pain:
        console.print(f"  Max Pain: ${profile.max_pain:.2f}")
    if profile.support_levels:
        console.print(f"  Support: {', '.join(f'${s:.0f}' for s in profile.support_levels[:3])}")
    if profile.resistance_levels:
        console.print(
            f"  Resistance: {', '.join(f'${r:.0f}' for r in profile.resistance_levels[:3])}"
        )
    console.print(
        f"  Lotto favorable: {'Yes' if profile.lotto_favorable else 'No'}"
    )
    console.print(f"  Score: {profile.strength:.1f}/10")
    console.print(f"  {profile.rationale}")


@scanner_app.command("portfolio")
def portfolio_cmd(
    log_level: Annotated[str, typer.Option(help="Log level")] = "WARNING",
) -> None:
    """Show paper trading portfolio."""
    from flowedge.scanner.paper_trading.engine import AlpacaPaperTrader

    setup_logging(log_level)
    trader = AlpacaPaperTrader()
    portfolio = asyncio.run(trader.get_portfolio())

    console.print("\n[bold]Paper Portfolio[/bold]")
    console.print(f"  Cash: ${portfolio.cash:,.2f}")
    console.print(f"  Portfolio value: ${portfolio.portfolio_value:,.2f}")
    console.print(f"  P&L: ${portfolio.total_pnl:+,.2f} ({portfolio.total_pnl_pct:+.2f}%)")

    if portfolio.positions:
        table = Table(title="Positions")
        table.add_column("Ticker")
        table.add_column("Qty", justify="right")
        table.add_column("Avg Entry", justify="right")
        table.add_column("Current", justify="right")
        table.add_column("P&L", justify="right")
        table.add_column("P&L %", justify="right")
        for p in portfolio.positions:
            pnl_color = "green" if p.unrealized_pnl >= 0 else "red"
            table.add_row(
                p.ticker,
                str(p.qty),
                f"${p.avg_entry:.2f}",
                f"${p.current_price:.2f}",
                f"[{pnl_color}]${p.unrealized_pnl:+,.2f}[/{pnl_color}]",
                f"[{pnl_color}]{p.unrealized_pnl_pct:+.1f}%[/{pnl_color}]",
            )
        console.print(table)
    else:
        console.print("  [yellow]No open positions[/yellow]")


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
