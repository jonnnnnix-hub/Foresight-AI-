"""Base class for all council specialists."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from datetime import date

from flowedge.council.models import SpecialistReview
from flowedge.scanner.backtest.scalp_config import ScalpConfig
from flowedge.scanner.backtest.schemas import BacktestResult


class BaseSpecialist(ABC):
    """Every specialist must implement `analyze()`.

    Receives the latest backtest result plus optional historical results
    for trend analysis, and returns a structured SpecialistReview.
    """

    name: str = "Base Specialist"
    specialist_id: str = "base"

    def run(
        self,
        result: BacktestResult,
        config: ScalpConfig,
        review_date: date,
        history: list[BacktestResult] | None = None,
    ) -> SpecialistReview:
        """Execute analysis with timing. Delegates to `analyze()`."""
        t0 = time.perf_counter()
        review = self.analyze(result, config, review_date, history or [])
        review.computation_time_ms = round((time.perf_counter() - t0) * 1000, 1)
        review.specialist_name = self.name
        review.specialist_id = self.specialist_id
        review.review_date = review_date
        return review

    @abstractmethod
    def analyze(
        self,
        result: BacktestResult,
        config: ScalpConfig,
        review_date: date,
        history: list[BacktestResult],
    ) -> SpecialistReview:
        """Perform the specialist's analysis. Must be implemented by subclasses."""
        ...
