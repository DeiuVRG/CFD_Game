import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import List

logger = logging.getLogger(__name__)


@dataclass
class TradeRecord:
    timestamp: datetime
    epic: str
    direction: str
    size: float
    entry_price: float
    strategy: str
    deal_id: str
    pnl: float = 0.0
    closed: bool = False


@dataclass
class PerformanceSummary:
    total_trades: int = 0
    wins: int = 0
    losses: int = 0
    total_pnl: float = 0.0
    current_equity: float = 0.0
    day_start_equity: float = 0.0
    day_pnl: float = 0.0
    open_positions: int = 0

    @property
    def win_rate(self) -> float:
        closed = self.wins + self.losses
        if closed == 0:
            return 0.0
        return self.wins / closed * 100


class PerformanceTracker:
    def __init__(self):
        self.trades: List[TradeRecord] = []
        self.equity_snapshots: List[tuple] = []
        self._day_start_equity: float = 0.0
        self._current_equity: float = 0.0

    def record_entry(self, epic: str, direction: str, size: float,
                     entry_price: float, strategy: str, deal_id: str):
        self.trades.append(TradeRecord(
            timestamp=datetime.utcnow(),
            epic=epic, direction=direction, size=size,
            entry_price=entry_price, strategy=strategy,
            deal_id=deal_id,
        ))

    def update_equity(self, equity: float):
        self._current_equity = equity
        self.equity_snapshots.append((datetime.utcnow(), equity))

        if self._day_start_equity == 0.0:
            self._day_start_equity = equity

    def get_summary(self, open_positions_count: int = 0) -> PerformanceSummary:
        wins = sum(1 for t in self.trades if t.closed and t.pnl > 0)
        losses = sum(1 for t in self.trades if t.closed and t.pnl <= 0)
        total_pnl = sum(t.pnl for t in self.trades if t.closed)

        day_pnl = self._current_equity - self._day_start_equity
        if self._day_start_equity > 0:
            day_pnl = self._current_equity - self._day_start_equity
        else:
            day_pnl = 0.0

        return PerformanceSummary(
            total_trades=len(self.trades),
            wins=wins,
            losses=losses,
            total_pnl=total_pnl,
            current_equity=self._current_equity,
            day_start_equity=self._day_start_equity,
            day_pnl=day_pnl,
            open_positions=open_positions_count,
        )
