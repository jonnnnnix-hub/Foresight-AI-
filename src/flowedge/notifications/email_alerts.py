"""Email trade alerts — push notifications on every order.

Sends HTML-formatted emails via SMTP on:
- Trade entry (buy signal fired)
- Trade exit (TP / trailing stop / time exit / stop loss)
- Scanner errors (optional)

Configuration via .env:
    ALERT_EMAIL_TO=jonnnnnix@gmail.com
    ALERT_EMAIL_FROM=flowedge.alerts@gmail.com
    ALERT_SMTP_HOST=smtp.gmail.com
    ALERT_SMTP_PORT=587
    ALERT_SMTP_USER=flowedge.alerts@gmail.com
    ALERT_SMTP_PASS=your-app-password

Usage:
    from flowedge.notifications.email_alerts import send_trade_alert
    await send_trade_alert(trade_data)
"""

from __future__ import annotations

import asyncio
import os
import smtplib
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Any

import structlog

logger = structlog.get_logger()

# ── Config from env ──────────────────────────────────────────────

ALERT_EMAIL_TO = os.getenv("ALERT_EMAIL_TO", "jonnnnnix@gmail.com")
ALERT_EMAIL_FROM = os.getenv("ALERT_EMAIL_FROM", "")
ALERT_SMTP_HOST = os.getenv("ALERT_SMTP_HOST", "smtp.gmail.com")
ALERT_SMTP_PORT = int(os.getenv("ALERT_SMTP_PORT", "587"))
ALERT_SMTP_USER = os.getenv("ALERT_SMTP_USER", "")
ALERT_SMTP_PASS = os.getenv("ALERT_SMTP_PASS", "")


def _is_configured() -> bool:
    """Check if SMTP credentials are set."""
    return bool(ALERT_EMAIL_FROM and ALERT_SMTP_USER and ALERT_SMTP_PASS)


def _send_email(subject: str, html_body: str) -> bool:
    """Send an HTML email via SMTP. Returns True on success."""
    if not _is_configured():
        logger.warning(
            "email_not_configured",
            hint="Set ALERT_EMAIL_FROM, ALERT_SMTP_USER, ALERT_SMTP_PASS "
            "in .env to enable email alerts",
        )
        return False

    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"] = ALERT_EMAIL_FROM
        msg["To"] = ALERT_EMAIL_TO

        # Plain text fallback
        plain = html_body.replace("<br>", "\n").replace("</td>", " | ")
        import re
        plain = re.sub(r"<[^>]+>", "", plain)
        msg.attach(MIMEText(plain, "plain"))
        msg.attach(MIMEText(html_body, "html"))

        with smtplib.SMTP(ALERT_SMTP_HOST, ALERT_SMTP_PORT) as server:
            server.starttls()
            server.login(ALERT_SMTP_USER, ALERT_SMTP_PASS)
            server.send_message(msg)

        logger.info(
            "email_sent",
            subject=subject,
            to=ALERT_EMAIL_TO,
        )
        return True

    except Exception as e:
        logger.error(
            "email_send_failed",
            error=str(e),
            subject=subject,
        )
        return False


# ── Trade Alert Templates ────────────────────────────────────────

def _entry_html(trade: dict[str, Any]) -> str:
    """Build HTML for a trade entry alert."""
    model = trade.get("model", "unknown")
    ticker = trade.get("ticker", "?")
    conviction = trade.get("conviction", 0)
    contract = trade.get("contract", "?")
    strike = trade.get("strike", 0)
    premium = trade.get("premium", 0)
    contracts = trade.get("contracts", 0)
    cost = premium * contracts * 100
    signal = trade.get("signal", {})

    color = "#22c55e"  # green for entry
    return f"""
    <div style="font-family: -apple-system, sans-serif; max-width: 600px; margin: 0 auto; background: #0f1117; color: #e5e7eb; padding: 24px; border-radius: 12px;">
        <div style="background: {color}; color: #000; padding: 12px 20px; border-radius: 8px; margin-bottom: 16px;">
            <h2 style="margin: 0; font-size: 18px;">BUY — {ticker} ({model})</h2>
        </div>
        <table style="width: 100%; border-collapse: collapse; font-size: 14px;">
            <tr><td style="padding: 8px 0; color: #9ca3af;">Model</td><td style="padding: 8px 0; text-align: right; font-weight: bold;">{model}</td></tr>
            <tr><td style="padding: 8px 0; color: #9ca3af;">Ticker</td><td style="padding: 8px 0; text-align: right; font-weight: bold;">{ticker}</td></tr>
            <tr><td style="padding: 8px 0; color: #9ca3af;">Conviction</td><td style="padding: 8px 0; text-align: right; font-weight: bold;">{conviction:.1f}/10</td></tr>
            <tr><td style="padding: 8px 0; color: #9ca3af;">Contract</td><td style="padding: 8px 0; text-align: right;">{contract}</td></tr>
            <tr><td style="padding: 8px 0; color: #9ca3af;">Strike</td><td style="padding: 8px 0; text-align: right;">${strike:.2f}</td></tr>
            <tr><td style="padding: 8px 0; color: #9ca3af;">Premium</td><td style="padding: 8px 0; text-align: right;">${premium:.2f}</td></tr>
            <tr><td style="padding: 8px 0; color: #9ca3af;">Contracts</td><td style="padding: 8px 0; text-align: right;">{contracts}</td></tr>
            <tr style="border-top: 1px solid #2a2d3a;"><td style="padding: 8px 0; color: #9ca3af;">Total Cost</td><td style="padding: 8px 0; text-align: right; font-weight: bold; color: {color};">${cost:,.2f}</td></tr>
        </table>
        {"<div style='margin-top: 16px; padding: 12px; background: #1a1d29; border-radius: 8px; font-size: 12px;'><strong>Signal Details:</strong><br>IBS=" + str(round(signal.get("ibs", 0), 4)) + " RSI3=" + str(round(signal.get("rsi3", 0), 2)) + " Vol=" + str(round(signal.get("vol_ratio", 0), 1)) + "x Drop=" + str(round(signal.get("intraday_drop_pct", 0) * 100, 2)) + "%</div>" if signal else ""}
        {"<div style='margin-top: 8px; padding: 12px; background: #1a1d29; border-radius: 8px; font-size: 12px;'><strong>FLUX:</strong> score=" + str(round(trade.get("flux_strength", 0), 1)) + " bias=" + str(trade.get("flux_bias", "n/a")) + " blocks=" + str(trade.get("flux_blocks", 0)) + "</div>" if trade.get("flux_strength") else ""}
        <div style="margin-top: 16px; font-size: 11px; color: #6b7280; text-align: center;">
            FlowEdge Scanner &mdash; {datetime.now().strftime("%Y-%m-%d %H:%M:%S ET")}
        </div>
    </div>
    """


def _exit_html(trade: dict[str, Any]) -> str:
    """Build HTML for a trade exit alert."""
    model = trade.get("model", "unknown")
    ticker = trade.get("ticker", "?")
    reason = trade.get("reason", "unknown")
    bars_held = trade.get("bars_held", 0)
    entry_price = trade.get("entry_price_option", 0)
    peak_price = trade.get("peak_option_price", 0)

    reason_labels = {
        "take_profit": ("TAKE PROFIT", "#22c55e"),
        "tp": ("TAKE PROFIT", "#22c55e"),
        "trailing_stop": ("TRAILING STOP", "#eab308"),
        "trail": ("TRAILING STOP", "#eab308"),
        "time_exit": ("TIME EXIT", "#3b82f6"),
        "stop": ("STOP LOSS", "#ef4444"),
        "emergency_stop": ("EMERGENCY STOP", "#ef4444"),
        "flux_reversal": ("FLUX FLOW REVERSAL", "#f59e0b"),
        "flux_flow_reversal": ("FLUX FLOW REVERSAL", "#f59e0b"),
    }
    label, color = reason_labels.get(reason, (reason.upper(), "#9ca3af"))

    return f"""
    <div style="font-family: -apple-system, sans-serif; max-width: 600px; margin: 0 auto; background: #0f1117; color: #e5e7eb; padding: 24px; border-radius: 12px;">
        <div style="background: {color}; color: #000; padding: 12px 20px; border-radius: 8px; margin-bottom: 16px;">
            <h2 style="margin: 0; font-size: 18px;">EXIT — {ticker} ({model}) — {label}</h2>
        </div>
        <table style="width: 100%; border-collapse: collapse; font-size: 14px;">
            <tr><td style="padding: 8px 0; color: #9ca3af;">Model</td><td style="padding: 8px 0; text-align: right; font-weight: bold;">{model}</td></tr>
            <tr><td style="padding: 8px 0; color: #9ca3af;">Ticker</td><td style="padding: 8px 0; text-align: right; font-weight: bold;">{ticker}</td></tr>
            <tr><td style="padding: 8px 0; color: #9ca3af;">Exit Reason</td><td style="padding: 8px 0; text-align: right; color: {color}; font-weight: bold;">{label}</td></tr>
            <tr><td style="padding: 8px 0; color: #9ca3af;">Bars Held</td><td style="padding: 8px 0; text-align: right;">{bars_held} ({bars_held * 5} min)</td></tr>
            <tr><td style="padding: 8px 0; color: #9ca3af;">Entry Premium</td><td style="padding: 8px 0; text-align: right;">${entry_price:.2f}</td></tr>
            <tr><td style="padding: 8px 0; color: #9ca3af;">Peak Premium</td><td style="padding: 8px 0; text-align: right;">${peak_price:.2f}</td></tr>
        </table>
        <div style="margin-top: 16px; font-size: 11px; color: #6b7280; text-align: center;">
            FlowEdge Scanner &mdash; {datetime.now().strftime("%Y-%m-%d %H:%M:%S ET")}
        </div>
    </div>
    """


# ── Public API ───────────────────────────────────────────────────

async def send_trade_entry_alert(trade: dict[str, Any]) -> bool:
    """Send email alert for a new trade entry. Non-blocking."""
    model = trade.get("model", "unknown")
    ticker = trade.get("ticker", "?")
    conviction = trade.get("conviction", 0)

    subject = f"🟢 BUY {ticker} — {model} (conviction {conviction:.0f})"
    html = _entry_html(trade)

    # Run in thread to avoid blocking the scan loop
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _send_email, subject, html)


async def send_trade_exit_alert(trade: dict[str, Any]) -> bool:
    """Send email alert for a trade exit. Non-blocking."""
    model = trade.get("model", "unknown")
    ticker = trade.get("ticker", "?")
    reason = trade.get("reason", "exit")

    emoji_map = {
        "take_profit": "💰", "tp": "💰",
        "trailing_stop": "⚡", "trail": "⚡",
        "time_exit": "⏰",
        "stop": "🔴", "emergency_stop": "🔴",
    }
    emoji = emoji_map.get(reason, "📤")

    subject = f"{emoji} EXIT {ticker} — {model} ({reason})"
    html = _exit_html(trade)

    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _send_email, subject, html)


async def send_scanner_alert(
    title: str,
    message: str,
    model: str = "",
) -> bool:
    """Send a generic scanner alert (errors, warnings, etc.)."""
    subject = f"⚠️ FlowEdge {model}: {title}"
    html = f"""
    <div style="font-family: -apple-system, sans-serif; max-width: 600px; margin: 0 auto; background: #0f1117; color: #e5e7eb; padding: 24px; border-radius: 12px;">
        <div style="background: #eab308; color: #000; padding: 12px 20px; border-radius: 8px; margin-bottom: 16px;">
            <h2 style="margin: 0; font-size: 18px;">{title}</h2>
        </div>
        <p style="font-size: 14px; line-height: 1.6;">{message}</p>
        <div style="margin-top: 16px; font-size: 11px; color: #6b7280; text-align: center;">
            FlowEdge Scanner &mdash; {datetime.now().strftime("%Y-%m-%d %H:%M:%S ET")}
        </div>
    </div>
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _send_email, subject, html)
