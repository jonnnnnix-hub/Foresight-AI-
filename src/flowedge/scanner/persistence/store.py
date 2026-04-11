"""JSON file-based persistence for watchlists, scan history, and tracking.

Uses flat JSON files for MVP. Migrate to PostgreSQL when needed.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import structlog

from flowedge.scanner.persistence.schemas import (
    LearningStats,
    ScanHistoryEntry,
    TrackedAlert,
    TradeStatus,
    Watchlist,
    WatchlistItem,
)
from flowedge.scanner.schemas.signals import LottoOpportunity, ScannerResult

logger = structlog.get_logger()

DEFAULT_DATA_DIR = Path("./data/scanner")


class ScannerStore:
    """File-based persistence for scanner state."""

    def __init__(self, data_dir: Path = DEFAULT_DATA_DIR) -> None:
        self._dir = data_dir
        self._dir.mkdir(parents=True, exist_ok=True)

    def _path(self, name: str) -> Path:
        return self._dir / f"{name}.json"

    def _read(self, name: str) -> Any:
        path = self._path(name)
        if not path.exists():
            return None
        return json.loads(path.read_text())

    def _write(self, name: str, data: Any) -> None:
        path = self._path(name)
        path.write_text(json.dumps(data, indent=2, default=str))

    # ---- Watchlist ----

    def get_watchlist(self, name: str = "default") -> Watchlist:
        data = self._read(f"watchlist_{name}")
        if data:
            return Watchlist.model_validate(data)
        return Watchlist(name=name)

    def save_watchlist(self, watchlist: Watchlist) -> None:
        self._write(
            f"watchlist_{watchlist.name}",
            watchlist.model_dump(mode="json"),
        )
        logger.info("watchlist_saved", name=watchlist.name, count=len(watchlist.items))

    def add_to_watchlist(
        self,
        ticker: str,
        name: str = "default",
        notes: str = "",
        tags: list[str] | None = None,
    ) -> Watchlist:
        wl = self.get_watchlist(name)
        if ticker.upper() not in [i.ticker for i in wl.items]:
            wl.items.append(
                WatchlistItem(
                    ticker=ticker.upper(),
                    notes=notes,
                    tags=tags or [],
                )
            )
            wl.updated_at = datetime.now()
            self.save_watchlist(wl)
        return wl

    def remove_from_watchlist(self, ticker: str, name: str = "default") -> Watchlist:
        wl = self.get_watchlist(name)
        wl.items = [i for i in wl.items if i.ticker != ticker.upper()]
        wl.updated_at = datetime.now()
        self.save_watchlist(wl)
        return wl

    # ---- Scan History ----

    def log_scan(self, result: ScannerResult) -> ScanHistoryEntry:
        top = result.top_opportunities
        entry = ScanHistoryEntry(
            scan_id=result.scan_id,
            scanned_at=result.scanned_at,
            tickers_scanned=result.tickers_scanned,
            opportunities_found=len(result.opportunities),
            top_ticker=top[0].ticker if top else "",
            top_score=top[0].composite_score if top else 0.0,
        )

        # Append to history log
        history = self._read("scan_history") or []
        history.append(entry.model_dump(mode="json"))
        # Keep last 1000 entries
        self._write("scan_history", history[-1000:])
        return entry

    def get_scan_history(self, limit: int = 50) -> list[ScanHistoryEntry]:
        data = self._read("scan_history") or []
        entries = [ScanHistoryEntry.model_validate(d) for d in data]
        return entries[-limit:]

    # ---- Alert Tracking ----

    def track_alert(self, opp: LottoOpportunity, scan_id: str = "") -> TrackedAlert:
        alert = TrackedAlert(
            alert_id=str(uuid.uuid4())[:12],
            ticker=opp.ticker,
            scan_id=scan_id,
            composite_score=opp.composite_score,
            uoa_score=opp.uoa_score,
            iv_score=opp.iv_score,
            catalyst_score=opp.catalyst_score,
            direction=opp.suggested_direction.value,
        )

        alerts = self._read("tracked_alerts") or []
        alerts.append(alert.model_dump(mode="json"))
        self._write("tracked_alerts", alerts[-5000:])

        logger.info("alert_tracked", alert_id=alert.alert_id, ticker=alert.ticker)
        return alert

    def update_alert(
        self,
        alert_id: str,
        status: TradeStatus | None = None,
        entry_price: float | None = None,
        exit_price: float | None = None,
        pnl_pct: float | None = None,
        notes: str = "",
    ) -> TrackedAlert | None:
        alerts = self._read("tracked_alerts") or []
        for i, data in enumerate(alerts):
            if data.get("alert_id") == alert_id:
                if status:
                    data["status"] = status.value
                if entry_price is not None:
                    data["entry_price"] = entry_price
                    data["entered_at"] = datetime.now().isoformat()
                if exit_price is not None:
                    data["exit_price"] = exit_price
                    data["exited_at"] = datetime.now().isoformat()
                if pnl_pct is not None:
                    data["pnl_pct"] = pnl_pct
                    data["was_winner"] = pnl_pct > 0
                if notes:
                    data["notes"] = notes
                alerts[i] = data
                self._write("tracked_alerts", alerts)
                return TrackedAlert.model_validate(data)
        return None

    def get_tracked_alerts(
        self,
        status: TradeStatus | None = None,
        limit: int = 100,
    ) -> list[TrackedAlert]:
        data = self._read("tracked_alerts") or []
        alerts = [TrackedAlert.model_validate(d) for d in data]
        if status:
            alerts = [a for a in alerts if a.status == status]
        return alerts[-limit:]

    # ---- Learning Feedback ----

    def compute_learning_stats(self) -> LearningStats:
        """Analyze alert history to find what works."""
        alerts = self.get_tracked_alerts()
        resolved = [
            a for a in alerts
            if a.status in (TradeStatus.EXITED, TradeStatus.EXPIRED)
            and a.pnl_pct is not None
        ]

        if not resolved:
            return LearningStats(total_alerts=len(alerts))

        winners = [a for a in resolved if a.was_winner]
        losers = [a for a in resolved if not a.was_winner]

        gross_profit = sum(a.pnl_pct or 0 for a in winners)
        gross_loss = abs(sum(a.pnl_pct or 0 for a in losers))

        # By score bucket
        buckets: dict[str, list[TrackedAlert]] = {
            "0-3": [],
            "3-5": [],
            "5-7": [],
            "7-10": [],
        }
        for a in resolved:
            if a.composite_score < 3:
                buckets["0-3"].append(a)
            elif a.composite_score < 5:
                buckets["3-5"].append(a)
            elif a.composite_score < 7:
                buckets["5-7"].append(a)
            else:
                buckets["7-10"].append(a)

        by_score: dict[str, dict[str, float]] = {}
        best_bucket = ""
        best_wr = 0.0
        for bname, blist in buckets.items():
            if not blist:
                continue
            bw = sum(1 for a in blist if a.was_winner)
            wr = bw / len(blist)
            by_score[bname] = {
                "count": float(len(blist)),
                "win_rate": round(wr, 3),
                "avg_pnl": round(
                    sum(a.pnl_pct or 0 for a in blist) / len(blist), 2
                ),
            }
            if wr > best_wr:
                best_wr = wr
                best_bucket = bname

        entered = [a for a in alerts if a.status != TradeStatus.ALERT]

        return LearningStats(
            total_alerts=len(alerts),
            total_entered=len(entered),
            total_resolved=len(resolved),
            winners=len(winners),
            losers=len(losers),
            win_rate=round(len(winners) / len(resolved), 3) if resolved else 0.0,
            avg_winner_pnl_pct=(
                round(gross_profit / len(winners), 2) if winners else 0.0
            ),
            avg_loser_pnl_pct=(
                round(-gross_loss / len(losers), 2) if losers else 0.0
            ),
            profit_factor=(
                round(gross_profit / gross_loss, 2) if gross_loss > 0 else 0.0
            ),
            best_score_bucket=best_bucket,
            by_score_bucket=by_score,
        )
