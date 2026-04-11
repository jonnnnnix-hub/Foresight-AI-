"""Scheduled scanner — runs periodically and sends alerts.

Supports alert channels: console, webhook (Slack/Discord), and file logging.
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx
import structlog

from flowedge.config.settings import Settings, get_settings
from flowedge.scanner.catalyst.engine import scan_catalysts
from flowedge.scanner.iv_rank.engine import scan_iv
from flowedge.scanner.providers.registry import ProviderRegistry
from flowedge.scanner.schemas.signals import LottoOpportunity, ScannerResult
from flowedge.scanner.scorer.engine import score_lottos
from flowedge.scanner.selector.engine import select_contracts
from flowedge.scanner.uoa.engine import scan_uoa

logger = structlog.get_logger()


class AlertChannel:
    """Base alert channel."""

    async def send(self, message: str, data: dict[str, Any]) -> None:
        raise NotImplementedError


class ConsoleAlert(AlertChannel):
    """Print alerts to console."""

    async def send(self, message: str, data: dict[str, Any]) -> None:
        print(f"\n🔔 ALERT: {message}")
        for key, val in data.items():
            print(f"  {key}: {val}")


class WebhookAlert(AlertChannel):
    """Send alerts to a webhook (Slack, Discord, etc.)."""

    def __init__(self, webhook_url: str) -> None:
        self._url = webhook_url

    async def send(self, message: str, data: dict[str, Any]) -> None:
        payload = {
            "text": message,
            "blocks": [
                {"type": "header", "text": {"type": "plain_text", "text": message}},
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": "\n".join(
                            f"*{k}*: {v}" for k, v in data.items()
                        ),
                    },
                },
            ],
        }
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                await client.post(self._url, json=payload)
        except Exception as e:
            logger.warning("webhook_alert_failed", error=str(e))


class FileAlert(AlertChannel):
    """Append alerts to a JSON lines file."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._path.parent.mkdir(parents=True, exist_ok=True)

    async def send(self, message: str, data: dict[str, Any]) -> None:
        entry = {
            "timestamp": datetime.now().isoformat(),
            "message": message,
            **data,
        }
        with open(self._path, "a") as f:
            f.write(json.dumps(entry, default=str) + "\n")


async def _format_alert(
    opp: LottoOpportunity,
    contracts_info: str = "",
) -> tuple[str, dict[str, Any]]:
    """Format a lotto opportunity into an alert message."""
    msg = f"🎯 {opp.ticker} — Lotto Score {opp.composite_score:.1f}/10"

    data: dict[str, Any] = {
        "ticker": opp.ticker,
        "composite_score": opp.composite_score,
        "direction": opp.suggested_direction.value,
        "uoa_score": opp.uoa_score,
        "iv_score": opp.iv_score,
        "catalyst_score": opp.catalyst_score,
    }

    if opp.entry_criteria:
        data["entry_criteria"] = " | ".join(opp.entry_criteria)
    if opp.risk_flags:
        data["risk_flags"] = " | ".join(opp.risk_flags)
    if contracts_info:
        data["suggested_contracts"] = contracts_info

    return msg, data


async def run_scheduled_scan(
    tickers: list[str],
    min_score: float = 5.0,
    channels: list[AlertChannel] | None = None,
    with_contracts: bool = True,
    settings: Settings | None = None,
) -> ScannerResult:
    """Run a single scan cycle and send alerts for qualifying opportunities."""
    settings = settings or get_settings()
    registry = ProviderRegistry(settings)
    channels = channels or [ConsoleAlert()]

    try:
        # Run all scanners
        uoa_signals = await scan_uoa(registry, tickers, settings)
        iv_signals = await scan_iv(registry, tickers, settings)
        catalyst_signals = await scan_catalysts(registry, tickers, settings)

        result = score_lottos(uoa_signals, iv_signals, catalyst_signals, settings)

        # Filter to qualifying opportunities
        qualified = [
            o for o in result.top_opportunities if o.composite_score >= min_score
        ]

        logger.info(
            "scan_cycle_complete",
            total=len(result.opportunities),
            qualified=len(qualified),
            min_score=min_score,
        )

        # Send alerts for new high-score opportunities
        for opp in qualified:
            contracts_info = ""
            if with_contracts:
                try:
                    chain_provider = registry.get_options_chain_provider()
                    chain = await chain_provider.get_options_chain(opp.ticker)
                    recs = select_contracts(opp, chain)
                    if recs:
                        contracts_info = " | ".join(
                            f"{r.contract.option_type.value} {r.contract.strike} "
                            f"exp={r.contract.expiration} "
                            f"RR={r.risk_reward:.1f}x"
                            for r in recs[:2]
                        )
                except Exception as e:
                    logger.debug("contract_select_failed", ticker=opp.ticker, error=str(e))

            msg, data = await _format_alert(opp, contracts_info)
            for channel in channels:
                try:
                    await channel.send(msg, data)
                except Exception as e:
                    logger.warning(
                        "alert_send_failed",
                        channel=type(channel).__name__,
                        error=str(e),
                    )

        return result

    finally:
        await registry.close_all()


async def run_scanner_loop(
    tickers: list[str],
    interval_minutes: int = 15,
    min_score: float = 5.0,
    channels: list[AlertChannel] | None = None,
    max_cycles: int | None = None,
    settings: Settings | None = None,
) -> None:
    """Run the scanner on a recurring interval.

    Args:
        tickers: Tickers to scan each cycle.
        interval_minutes: Minutes between scan cycles.
        min_score: Minimum composite score to trigger alert.
        channels: Alert channels (console, webhook, file).
        max_cycles: Stop after N cycles (None = run forever).
        settings: Override settings.
    """
    cycle = 0
    logger.info(
        "scanner_loop_started",
        tickers=tickers,
        interval=f"{interval_minutes}m",
        min_score=min_score,
    )

    while max_cycles is None or cycle < max_cycles:
        cycle += 1
        logger.info("scan_cycle_start", cycle=cycle)

        try:
            await run_scheduled_scan(
                tickers=tickers,
                min_score=min_score,
                channels=channels,
                settings=settings,
            )
        except Exception as e:
            logger.error("scan_cycle_error", cycle=cycle, error=str(e))

        if max_cycles is not None and cycle >= max_cycles:
            break

        logger.info(
            "scan_cycle_sleeping",
            next_in=f"{interval_minutes}m",
        )
        await asyncio.sleep(interval_minutes * 60)

    logger.info("scanner_loop_stopped", total_cycles=cycle)
