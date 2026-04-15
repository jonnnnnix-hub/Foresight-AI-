"""Backtest run history — persists every backtest run for comparison over time.

Every call to run_shares_backtest() or run_scalp_real_backtest() produces a
BacktestResult. This module stores each run with a timestamp, model name,
and parameters so you can compare performance across runs and over time.

Storage: data/run_history/runs.jsonl (append-only, one JSON object per line)
Index:   data/run_history/index.json (summary for quick dashboard loading)

The JSONL format avoids rewriting the entire file on each append.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import structlog

from flowedge.scanner.backtest.schemas import BacktestResult

logger = structlog.get_logger()

HISTORY_DIR = Path("data/run_history")
RUNS_FILE = HISTORY_DIR / "runs.jsonl"
INDEX_FILE = HISTORY_DIR / "index.json"


def record_run(
    result: BacktestResult,
    model_name: str,
    params: dict[str, Any] | None = None,
    tags: list[str] | None = None,
) -> str:
    """Persist a backtest run to the history log.

    Args:
        result: The completed backtest result.
        model_name: e.g. "precision_shares", "scalp_real", "ensemble"
        params: The parameter overrides used (None = defaults).
        tags: Optional tags like "optimizer", "specialist", "manual".

    Returns:
        The run_id of the persisted entry.
    """
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)

    entry: dict[str, Any] = {
        "run_id": result.run_id,
        "timestamp": datetime.now().isoformat(),
        "model_name": model_name,
        "params": params or {},
        "tags": tags or [],
        "tickers": result.tickers,
        "total_trades": result.total_trades,
        "wins": result.wins,
        "losses": result.losses,
        "win_rate": result.win_rate,
        "avg_win_pct": result.avg_win_pct,
        "avg_loss_pct": result.avg_loss_pct,
        "total_pnl_pct": result.total_pnl_pct,
        "profit_factor": result.profit_factor,
        "avg_hold_days": result.avg_hold_days,
        "expectancy_pct": result.expectancy_pct,
        "starting_capital": result.starting_capital,
        "ending_value": result.ending_value,
        "portfolio_return_pct": result.portfolio_return_pct,
        "max_drawdown_pct": result.max_drawdown_pct,
        "sharpe_ratio": result.sharpe_ratio,
        "by_ticker": result.by_ticker,
    }

    # Append to JSONL
    with RUNS_FILE.open("a") as f:
        f.write(json.dumps(entry, default=str) + "\n")

    # Update index
    _update_index(entry)

    logger.info(
        "run_recorded",
        run_id=result.run_id,
        model=model_name,
        trades=result.total_trades,
        wr=result.win_rate,
        ret=result.portfolio_return_pct,
    )

    return result.run_id


def record_full_run(
    result: BacktestResult,
    model_name: str,
    params: dict[str, Any] | None = None,
    tags: list[str] | None = None,
) -> str:
    """Persist a backtest run INCLUDING all individual trades.

    Use this for milestone runs you want to analyze in detail later.
    Saved to a separate per-run file to avoid bloating the JSONL.
    """
    run_id = record_run(result, model_name, params, tags)

    # Save full result with trades to a separate file
    detail_dir = HISTORY_DIR / "details"
    detail_dir.mkdir(parents=True, exist_ok=True)

    detail_path = detail_dir / f"{run_id}.json"
    detail_path.write_text(json.dumps(
        result.model_dump(mode="json"),
        indent=2,
        default=str,
    ))

    return run_id


def _update_index(entry: dict[str, Any]) -> None:
    """Update the summary index with a new entry."""
    index: dict[str, Any] = {"runs": [], "models": {}}
    if INDEX_FILE.exists():
        index = json.loads(INDEX_FILE.read_text())

    # Append summary
    summary = {
        "run_id": entry["run_id"],
        "timestamp": entry["timestamp"],
        "model_name": entry["model_name"],
        "tickers_count": len(entry["tickers"]),
        "total_trades": entry["total_trades"],
        "win_rate": entry["win_rate"],
        "portfolio_return_pct": entry["portfolio_return_pct"],
        "profit_factor": entry["profit_factor"],
        "max_drawdown_pct": entry["max_drawdown_pct"],
        "sharpe_ratio": entry["sharpe_ratio"],
        "tags": entry.get("tags", []),
    }
    index["runs"].append(summary)

    # Update per-model tracking
    model = entry["model_name"]
    if model not in index["models"]:
        index["models"][model] = {
            "runs": 0,
            "best_win_rate": 0,
            "best_return_pct": -999,
            "latest_win_rate": 0,
            "latest_return_pct": 0,
        }
    m = index["models"][model]
    m["runs"] += 1
    m["best_win_rate"] = max(m["best_win_rate"], entry["win_rate"])
    m["best_return_pct"] = max(m["best_return_pct"], entry["portfolio_return_pct"])
    m["latest_win_rate"] = entry["win_rate"]
    m["latest_return_pct"] = entry["portfolio_return_pct"]

    INDEX_FILE.write_text(json.dumps(index, indent=2, default=str))


# ── Query Functions ──────────────────────────────────────────────────────────


def load_run_history() -> list[dict[str, Any]]:
    """Load all run summaries from the JSONL file."""
    if not RUNS_FILE.exists():
        return []

    runs: list[dict[str, Any]] = []
    for line in RUNS_FILE.read_text().splitlines():
        line = line.strip()
        if line:
            runs.append(json.loads(line))

    return runs


def load_index() -> dict[str, Any]:
    """Load the summary index."""
    if not INDEX_FILE.exists():
        return {"runs": [], "models": {}}
    return json.loads(INDEX_FILE.read_text())


def get_model_timeline(model_name: str | None = None) -> list[dict[str, Any]]:
    """Get performance timeline for a specific model or all models.

    Returns list of {timestamp, model_name, win_rate, return_pct, ...}
    sorted by timestamp, suitable for charting.
    """
    runs = load_run_history()
    if model_name:
        runs = [r for r in runs if r["model_name"] == model_name]

    # Sort by timestamp
    runs.sort(key=lambda r: r.get("timestamp", ""))

    timeline: list[dict[str, Any]] = []
    for r in runs:
        timeline.append({
            "timestamp": r.get("timestamp", ""),
            "model_name": r.get("model_name", ""),
            "win_rate": r.get("win_rate", 0),
            "portfolio_return_pct": r.get("portfolio_return_pct", 0),
            "profit_factor": r.get("profit_factor", 0),
            "total_trades": r.get("total_trades", 0),
            "max_drawdown_pct": r.get("max_drawdown_pct", 0),
            "sharpe_ratio": r.get("sharpe_ratio", 0),
            "tickers_count": len(r.get("tickers", [])),
        })

    return timeline


def get_model_comparison() -> dict[str, list[dict[str, Any]]]:
    """Get performance timelines grouped by model name.

    Returns {model_name: [{timestamp, win_rate, return_pct, ...}, ...]}
    """
    runs = load_run_history()
    by_model: dict[str, list[dict[str, Any]]] = {}

    for r in runs:
        model = r.get("model_name", "unknown")
        if model not in by_model:
            by_model[model] = []
        by_model[model].append({
            "timestamp": r.get("timestamp", ""),
            "win_rate": r.get("win_rate", 0),
            "portfolio_return_pct": r.get("portfolio_return_pct", 0),
            "profit_factor": r.get("profit_factor", 0),
            "total_trades": r.get("total_trades", 0),
            "max_drawdown_pct": r.get("max_drawdown_pct", 0),
            "sharpe_ratio": r.get("sharpe_ratio", 0),
        })

    # Sort each model's entries by timestamp
    for entries in by_model.values():
        entries.sort(key=lambda e: e["timestamp"])

    return by_model


def get_available_models() -> list[str]:
    """Get list of all model names that have recorded runs."""
    index = load_index()
    return sorted(index.get("models", {}).keys())


# ── Backfill from existing files ─────────────────────────────────────────────


def backfill_from_existing() -> int:
    """Backfill run history from existing data/backtest/*.json files.

    Imports historical backtest results so the timeline chart has
    data from before the run_history module was introduced.
    """
    backtest_dir = Path("data/backtest")
    if not backtest_dir.exists():
        return 0

    existing_ids: set[str] = set()
    for run in load_run_history():
        existing_ids.add(run.get("run_id", ""))

    imported = 0
    for path in sorted(backtest_dir.glob("*.json")):
        try:
            data = json.loads(path.read_text())
            run_id = data.get("run_id", "")
            if run_id in existing_ids:
                continue

            result = BacktestResult(**data)

            # Infer model name from run_id or filename
            fname = path.stem.lower()
            if "precision" in fname or "precision" in run_id:
                model = "precision_shares"
            elif "hybrid" in fname or "hybrid" in run_id:
                model = "hybrid_shares"
            elif "scalp" in fname or "scalp" in run_id:
                model = "scalp_real"
            elif "rapid" in fname or "rapid" in run_id:
                model = "rapid_intraday"
            elif "ensemble" in fname:
                model = "ensemble"
            elif "index" in fname:
                model = "index_specialist"
            else:
                model = "unknown"

            record_run(result, model, tags=["backfill"])
            imported += 1
        except Exception as e:
            logger.debug("backfill_skip", path=str(path), error=str(e)[:100])

    logger.info("backfill_complete", imported=imported)
    return imported
