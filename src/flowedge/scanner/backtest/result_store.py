"""Persist and load backtest results to JSON files.

Results are saved to data/backtest_results/ with a timestamped filename.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from flowedge.scanner.backtest.schemas import BacktestResult

RESULTS_DIR = Path("data/backtest_results")


def save_result(result: BacktestResult, tag: str = "") -> Path:
    """Save a BacktestResult to a JSON file.

    Args:
        result: The backtest result to save.
        tag: Optional tag for the filename (e.g. "scalp-real", "walkforward").

    Returns:
        Path to the saved file.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag_part = f"_{tag}" if tag else ""
    filename = f"backtest{tag_part}_{ts}.json"
    path = RESULTS_DIR / filename

    # Use Pydantic's model_dump for clean serialization
    data = result.model_dump(mode="json")
    path.write_text(json.dumps(data, indent=2, default=str))
    return path


def load_result(path: Path) -> BacktestResult:
    """Load a BacktestResult from a JSON file."""
    data = json.loads(path.read_text())
    return BacktestResult.model_validate(data)


def list_results(tag: str = "") -> list[Path]:
    """List all saved backtest results, newest first (by mtime)."""
    if not RESULTS_DIR.exists():
        return []
    pattern = f"backtest_{tag}*.json" if tag else "backtest_*.json"
    return sorted(RESULTS_DIR.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
