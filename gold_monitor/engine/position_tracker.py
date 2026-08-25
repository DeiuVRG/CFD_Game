import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from config.settings import STRATEGY

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
    close_reason: Optional[str] = None
    # 24/7 instruments (crypto) are exempt from the end-of-day forced close
    eod_close: bool = True
    # Trailing SL state
    highest_price: float = 0.0    # For BUY positions
    lowest_price: float = 999999  # For SELL positions
    trailing_activated: bool = False
    original_sl: float = 0.0

    def __post_init__(self):
        self.original_sl = self.stop_loss
        if self.direction == "BUY":
            self.highest_price = self.entry_price
        else:
            self.lowest_price = self.entry_price

    @property
    def is_open(self) -> bool:
        return self.closed_at is None

    @property
    def pnl_pct(self) -> float:
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

    def update_trailing_sl(self, current_price: float, atr: float = 0):
        """Update trailing stop loss if enabled and conditions met."""
        if not STRATEGY.TRAILING_SL_ENABLED or atr <= 0:
            return

        pnl = self.unrealized_pnl_pct(current_price)

        # Activate trailing after minimum profit threshold
        if pnl < STRATEGY.TRAILING_SL_ACTIVATION * 100:
            return

        self.trailing_activated = True
        trail_dist = atr * STRATEGY.TRAILING_SL_DISTANCE

        if self.direction == "BUY":
            if current_price > self.highest_price:
                self.highest_price = current_price
            new_sl = self.highest_price - trail_dist
            if new_sl > self.stop_loss:
                self.stop_loss = new_sl
                logger.debug(f"Trailing SL updated: {self.instrument} SL -> {new_sl:.5f}")

        else:  # SELL
            if current_price < self.lowest_price:
                self.lowest_price = current_price
            new_sl = self.lowest_price + trail_dist
            if new_sl < self.stop_loss:
                self.stop_loss = new_sl
                logger.debug(f"Trailing SL updated: {self.instrument} SL -> {new_sl:.5f}")


class PositionTracker:
    """Tracks open positions with trailing SL and EOD close support."""

    def __init__(self):
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
                      stop_loss: float, take_profit: float, strategy_name: str,
                      eod_close: bool = True) -> TrackedPosition:
        if self.has_position(instrument):
            self.close_position(instrument, entry_price, "SIGNAL_REVERSED")

        pos = TrackedPosition(
            instrument=instrument,
            direction=direction,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            strategy_name=strategy_name,
            eod_close=eod_close,
        )
        self._positions[instrument] = pos
        logger.info(f"Position opened: {direction} {instrument} @ {entry_price}")
        return pos

    def close_position(self, instrument: str, close_price: float,
                       reason: str) -> Optional[TrackedPosition]:
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

    def check_sl_tp(self, instrument: str, current_price: float,
                    atr: float = 0) -> Optional[TrackedPosition]:
        """Check SL/TP. Also updates trailing SL."""
        pos = self.get_position(instrument)
        if pos is None:
            return None

        # Update trailing SL first
        pos.update_trailing_sl(current_price, atr)

        if pos.direction == "BUY":
            if current_price <= pos.stop_loss:
                reason = "TRAILING_SL_HIT" if pos.trailing_activated else "SL_HIT"
                return self.close_position(instrument, current_price, reason)
            if current_price >= pos.take_profit:
                return self.close_position(instrument, current_price, "TP_HIT")
        else:  # SELL
            if current_price >= pos.stop_loss:
                reason = "TRAILING_SL_HIT" if pos.trailing_activated else "SL_HIT"
                return self.close_position(instrument, current_price, reason)
            if current_price <= pos.take_profit:
                return self.close_position(instrument, current_price, "TP_HIT")

        return None

    def check_eod_close(self) -> list:
        """Close all positions at end of day. Returns list of closed positions."""
        now = datetime.utcnow()
        if now.hour < STRATEGY.EOD_CLOSE_UTC:
            return []

        closed = []
        for instrument in list(self._positions.keys()):
            pos = self.get_position(instrument)
            if pos is None:
                continue

            # 24/7 instruments (crypto) are never force-closed at EOD
            if not pos.eod_close:
                continue

            # Use last known price (entry as fallback)
            price = pos.entry_price  # Will be updated by engine before calling this
            result = self.close_position(instrument, price, "EOD_CLOSE")
            if result:
                closed.append(result)
                logger.info(f"EOD close: {instrument}")

        return closed

    def get_stats(self) -> dict:
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
