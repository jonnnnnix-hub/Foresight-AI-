"""Tests for scheduler alert channels."""

import json
from pathlib import Path

import pytest

from flowedge.scanner.scheduler.engine import ConsoleAlert, FileAlert


@pytest.mark.asyncio
async def test_console_alert(capsys: pytest.CaptureFixture[str]) -> None:
    alert = ConsoleAlert()
    await alert.send("Test alert", {"score": 8.0, "ticker": "TSLA"})
    captured = capsys.readouterr()
    assert "ALERT" in captured.out
    assert "TSLA" in captured.out


@pytest.mark.asyncio
async def test_file_alert(tmp_path: Path) -> None:
    log_file = tmp_path / "alerts.jsonl"
    alert = FileAlert(log_file)
    await alert.send("Test alert", {"score": 7.5, "ticker": "NVDA"})
    await alert.send("Second alert", {"score": 6.0, "ticker": "AAPL"})

    lines = log_file.read_text().strip().split("\n")
    assert len(lines) == 2

    first = json.loads(lines[0])
    assert first["ticker"] == "NVDA"
    assert first["message"] == "Test alert"
    assert "timestamp" in first
