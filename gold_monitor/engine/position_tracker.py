import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class TrackedPosition:
    """Represents an open position being tracked."""
    instrument: str       # e.g. "XAU/USD (Gold)"
    direction: str        # "BUY" or "SELL"
    entry_price: float
    stop_loss: float
    take_profit: float
    strategy_name: str
    opened_at: datetime = field(default_factory=datetime.utcnow)
    closed_at: Optional[datetime] = None
    close_price: Optional[float] = None
    close_reason: Optional[str] = None  # "TP_HIT", "SL_HIT", "SIGNAL_REVERSED"

    @property
    def is_open(self) -> bool:
        return self.closed_at is None

    @property
    def pnl_pct(self) -> float:
        """Calculate P&L percentage if closed."""
        if self.close_price is None:
            return 0.0
        if self.direction == "BUY":
            return (self.close_price - self.entry_price) / self.entry_price * 100
        else:
            return (self.entry_price - self.close_price) / self.entry_price * 100

    def unrealized_pnl_pct(self, current_price: float) -> float:
        if self.direction == "BUY":
            return (current_price - self.entry_price) / self.entry_price * 100
        else:
            return (self.entry_price - current_price) / self.entry_price * 100


class PositionTracker:
    """Tracks open positions per instrument and detects SL/TP hits."""

    def __init__(self):
        # instrument_display -> TrackedPosition
        self._positions: dict[str, TrackedPosition] = {}
        self._history: list[TrackedPosition] = []

    def has_position(self, instrument: str) -> bool:
        pos = self._positions.get(instrument)
        return pos is not None and pos.is_open

    def get_position(self, instrument: str) -> Optional[TrackedPosition]:
        pos = self._positions.get(instrument)
        if pos and pos.is_open:
            return pos
        return None

    def open_position(self, instrument: str, direction: str, entry_price: float,
                      stop_loss: float, take_profit: float, strategy_name: str) -> TrackedPosition:
        """Open a new tracked position (closes any existing one first)."""
        if self.has_position(instrument):
            self.close_position(instrument, entry_price, "SIGNAL_REVERSED")

        pos = TrackedPosition(
            instrument=instrument,
            direction=direction,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            strategy_name=strategy_name,
        )
        self._positions[instrument] = pos
        logger.info(f"Position opened: {direction} {instrument} @ {entry_price}")
        return pos

    def close_position(self, instrument: str, close_price: float,
                       reason: str) -> Optional[TrackedPosition]:
        """Close an existing position."""
        pos = self._positions.get(instrument)
        if pos is None or not pos.is_open:
            return None

        pos.closed_at = datetime.utcnow()
        pos.close_price = close_price
        pos.close_reason = reason
        self._history.append(pos)
        logger.info(
            f"Position closed: {pos.direction} {instrument} @ {close_price} "
            f"reason={reason} pnl={pos.pnl_pct:+.2f}%"
        )
        return pos

    def check_sl_tp(self, instrument: str, current_price: float) -> Optional[TrackedPosition]:
        """Check if current price hit SL or TP. Returns closed position if hit."""
        pos = self.get_position(instrument)
        if pos is None:
            return None

        if pos.direction == "BUY":
            if current_price <= pos.stop_loss:
                return self.close_position(instrument, current_price, "SL_HIT")
            if current_price >= pos.take_profit:
                return self.close_position(instrument, current_price, "TP_HIT")
        else:  # SELL
            if current_price >= pos.stop_loss:
                return self.close_position(instrument, current_price, "SL_HIT")
            if current_price <= pos.take_profit:
                return self.close_position(instrument, current_price, "TP_HIT")

        return None

    def get_stats(self) -> dict:
        """Get overall trading statistics."""
        if not self._history:
            return {"total": 0, "wins": 0, "losses": 0, "win_rate": 0.0}

        wins = sum(1 for p in self._history if p.pnl_pct > 0)
        losses = sum(1 for p in self._history if p.pnl_pct <= 0)
        total_pnl = sum(p.pnl_pct for p in self._history)

        return {
            "total": len(self._history),
            "wins": wins,
            "losses": losses,
            "win_rate": wins / len(self._history) if self._history else 0.0,
            "total_pnl_pct": total_pnl,
        }
