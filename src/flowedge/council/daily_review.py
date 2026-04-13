"""Daily post-market review pipeline.

Runs the council engine against the latest backtest results and
persists the review to disk for the dashboard to display.

Usage:
    from flowedge.council.daily_review import run_daily_review
    review = run_daily_review()  # Uses latest backtest result
    review = run_daily_review(config_path="configs/sweep_best_90wr.json")
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import structlog

from flowedge.council.council_engine import CouncilEngine
from flowedge.council.models import DailyReview, ReviewTrend
from flowedge.scanner.backtest.result_store import list_results, load_result
from flowedge.scanner.backtest.scalp_config import ScalpConfig
from flowedge.scanner.backtest.scalp_model_v2 import run_scalp_backtest_v2

logger = structlog.get_logger()

REVIEWS_DIR = Path("data/council_reviews")


def run_daily_review(
    config_path: str | Path | None = None,
    result_path: str | Path | None = None,
    run_backtest: bool = True,
    review_date: date | None = None,
) -> DailyReview:
    """Run the full council review pipeline.

    Args:
        config_path: Path to ScalpConfig JSON. Uses defaults if None.
        result_path: Path to a specific BacktestResult JSON. If None,
                     either runs a fresh backtest or loads the latest.
        run_backtest: If True and no result_path, run a fresh backtest.
        review_date: Date for the review. Defaults to today.

    Returns:
        The DailyReview object (also saved to disk).
    """
    review_date = review_date or date.today()

    # ── Load config ──────────────────────────────────────────────
    if config_path:
        config = ScalpConfig.from_json_file(config_path)
    else:
        config = ScalpConfig()

    # ── Get backtest result ──────────────────────────────────────
    if result_path:
        result = load_result(Path(result_path))
        logger.info("council_loaded_result", path=str(result_path))
    elif run_backtest:
        logger.info("council_running_backtest", tickers=len(config.tickers))
        result = run_scalp_backtest_v2(
            tickers=config.tickers,
            config=config,
            entry_mode="next_open",
            exit_mode="bar_close",
        )
        logger.info(
            "council_backtest_done",
            trades=result.total_trades,
            wr=result.win_rate,
        )
    else:
        # Load most recent result
        results = list_results()
        if not results:
            raise FileNotFoundError("No backtest results found. Run a backtest first.")
        result = load_result(results[0])
        logger.info("council_loaded_latest", path=str(results[0]))

    # ── Load historical results for trend analysis ───────────────
    history = _load_history(limit=10)

    # ── Run the council ──────────────────────────────────────────
    engine = CouncilEngine()
    review = engine.run_review(
        result=result,
        config=config,
        review_date=review_date,
        history=history,
    )

    # ── Persist review ───────────────────────────────────────────
    save_path = save_review(review)
    logger.info("council_review_saved", path=str(save_path))

    return review


def save_review(review: DailyReview) -> Path:
    """Save a DailyReview to a JSON file."""
    REVIEWS_DIR.mkdir(parents=True, exist_ok=True)
    filename = f"review_{review.review_date.isoformat()}_{review.review_id}.json"
    path = REVIEWS_DIR / filename
    path.write_text(review.model_dump_json(indent=2))
    return path


def load_review(path: Path) -> DailyReview:
    """Load a DailyReview from JSON."""
    data = json.loads(path.read_text())
    return DailyReview.model_validate(data)


def list_reviews() -> list[Path]:
    """List all saved council reviews, newest first."""
    if not REVIEWS_DIR.exists():
        return []
    return sorted(REVIEWS_DIR.glob("review_*.json"), reverse=True)


def get_review_trends(limit: int = 30) -> list[ReviewTrend]:
    """Load recent reviews and extract trend data for charts."""
    trends = []
    for path in list_reviews()[:limit]:
        try:
            review = load_review(path)
            top_finding = ""
            if review.specialist_reviews:
                # Get the most severe finding across all specialists
                all_findings = []
                for sr in review.specialist_reviews:
                    all_findings.extend(sr.findings)
                if all_findings:
                    all_findings.sort(
                        key=lambda f: (
                            {"critical": 0, "warning": 1, "info": 2}.get(f.severity.value, 3)
                        ),
                    )
                    top_finding = all_findings[0].title

            trends.append(
                ReviewTrend(
                    review_date=review.review_date,
                    status=review.status,
                    overall_health=review.overall_health,
                    trades=review.cumulative_trades,
                    win_rate=review.cumulative_wr,
                    pnl=review.cumulative_pnl,
                    top_finding=top_finding,
                )
            )
        except Exception as exc:
            logger.warning("council_trend_load_error", path=str(path), error=str(exc))
    return trends


def _load_history(limit: int = 10) -> list:
    """Load recent backtest results for historical context."""
    from flowedge.scanner.backtest.result_store import list_results, load_result

    history = []
    for path in list_results()[:limit]:
        try:
            history.append(load_result(path))
        except Exception:
            pass
    return history
