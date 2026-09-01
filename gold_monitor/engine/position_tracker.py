import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Callable, Optional

import pandas as pd

from config.settings import STRATEGY
from engine.execution_rules import EXIT_GAP_SL, EXIT_SL, EXIT_TP, v3_exit

logger = logging.getLogger(__name__)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _as_utc(ts) -> datetime:
    """Any timestamp (str / datetime / pandas) -> tz-aware UTC datetime."""
    t = pd.Timestamp(ts)
    t = t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")
    return t.to_pydatetime()


def floor_to_interval(ts: datetime, interval_hours: float) -> datetime:
    """Open time of the candle that contains `ts`."""
    step = int(round(interval_hours * 3600))
    epoch = int(_as_utc(ts).timestamp())
    return datetime.fromtimestamp(epoch - epoch % step, tz=timezone.utc)


# Outcome names persisted in signals.db / shown on Discord
OUTCOME_BY_EXIT = {
    EXIT_TP: "TP_HIT",
    EXIT_SL: "SL_HIT",
    EXIT_GAP_SL: "GAP_SL_HIT",
}


@dataclass
class TrackedPosition:
    """Represents an open (hypothetical) position being tracked."""
    instrument: str       # e.g. "XAU/USD (Gold)"
    direction: str        # "BUY" or "SELL"
    entry_price: float
    stop_loss: float
    take_profit: float
    strategy_name: str
    opened_at: datetime = field(default_factory=_utcnow)
    closed_at: Optional[datetime] = None
    close_price: Optional[float] = None
    close_reason: Optional[str] = None
    # 24/7 instruments (crypto) are exempt from the end-of-day forced close
    eod_close: bool = True
    # Link back to the persisted signal row + round-trip cost at entry (in %)
    signal_id: Optional[int] = None
    cost_pct: float = 0.0
    # Open time of the candle whose CLOSE produced the signal. Candle-based
    # outcome resolution (v3 rules) only looks at candles strictly after it -
    # the first of which is the entry candle.
    signal_candle_ts: Optional[datetime] = None

    @property
    def pnl_net_pct(self) -> float:
        """Hypothetical net P&L: gross minus the instrument's round-trip cost."""
        if self.close_price is None:
            return 0.0
        return self.pnl_pct - self.cost_pct
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
    """Tracks hypothetical open positions.

    Two resolution paths:
      - resolve_with_candles(): the v3 rules (wicks, SL first, gaps) applied
        candle by candle - the SAME rule the backtester validated. This is
        the evidence path; it is deterministic and replayable after a
        restart (restore_from_store()).
      - check_sl_tp() / check_eod_close(): tick-based checks with trailing SL
        and end-of-day close, kept for the legacy "vote" signal mode.

    When a SignalStore is attached, every close fills the hypothetical
    outcome (reason, exit price, gross and net P&L) of the originating
    signal exactly once.
    """

    def __init__(self, store=None):
        self._positions: dict[str, TrackedPosition] = {}
        self._history: list[TrackedPosition] = []
        self._store = store

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
                      eod_close: bool = True, signal_id: int = None,
                      cost_pct: float = 0.0,
                      signal_candle_ts: datetime = None,
                      opened_at: datetime = None) -> TrackedPosition:
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
            signal_id=signal_id,
            cost_pct=cost_pct,
            signal_candle_ts=_as_utc(signal_candle_ts) if signal_candle_ts else None,
        )
        if opened_at is not None:
            pos.opened_at = _as_utc(opened_at)
        self._positions[instrument] = pos
        logger.info(f"Position opened: {direction} {instrument} @ {entry_price}")
        return pos

    def close_position(self, instrument: str, close_price: float,
                       reason: str, closed_at: datetime = None) -> Optional[TrackedPosition]:
        pos = self._positions.get(instrument)
        if pos is None or not pos.is_open:
            return None
        return self._close(pos, close_price, reason, closed_at)

    def _close(self, pos: TrackedPosition, close_price: float, reason: str,
               closed_at: datetime = None) -> TrackedPosition:
        pos.closed_at = _as_utc(closed_at) if closed_at else _utcnow()
        pos.close_price = close_price
        pos.close_reason = reason
        self._history.append(pos)
        logger.info(
            f"Position closed: {pos.direction} {pos.instrument} @ {close_price} "
            f"reason={reason} pnl={pos.pnl_pct:+.2f}%"
        )

        # Fill the hypothetical outcome of the originating signal
        if self._store is not None and pos.signal_id is not None:
            try:
                self._store.record_outcome(
                    signal_id=pos.signal_id,
                    outcome=reason,
                    exit_price=close_price,
                    pnl_gross_pct=pos.pnl_pct,
                    pnl_net_pct=pos.pnl_net_pct,
                    ts_utc=pos.closed_at.strftime("%Y-%m-%dT%H:%M:%SZ"),
                )
            except Exception as e:
                logger.error(f"Failed to record signal outcome: {e}")

        return pos

    # ------------------------------------------------------------------
    # Candle-based resolution (v3 rules) - the evidence path
    # ------------------------------------------------------------------

    @staticmethod
    def _replay(pos: TrackedPosition, df: pd.DataFrame,
                until_ts: datetime = None):
        """Walk the candles strictly after pos.signal_candle_ts (or opened_at)
        and before `until_ts`; return (exit_price, outcome, candle_ts) of the
        first v3 exit, or None."""
        if df is None or df.empty or "timestamp" not in df.columns:
            return None
        ts = pd.to_datetime(df["timestamp"], utc=True)
        start = pos.signal_candle_ts or pos.opened_at
        mask = ts > pd.Timestamp(start)
        if until_ts is not None:
            mask &= ts < pd.Timestamp(_as_utc(until_ts))
        sub = df.loc[mask]
        for t, o, h, l in zip(ts[mask], sub["open"], sub["high"], sub["low"]):
            exit_price, why = v3_exit(pos.direction, pos.stop_loss,
                                      pos.take_profit, float(o), float(h), float(l))
            if exit_price is not None:
                return exit_price, OUTCOME_BY_EXIT[why], t.to_pydatetime()
        return None

    def resolve_with_candles(self, instrument: str,
                             df: pd.DataFrame) -> Optional[TrackedPosition]:
        """Apply the v3 exit rule to the completed candles after the signal
        candle. Returns the closed position, or None if still open."""
        pos = self.get_position(instrument)
        if pos is None:
            return None
        hit = self._replay(pos, df)
        if hit is None:
            return None
        exit_price, outcome, candle_ts = hit
        return self._close(pos, exit_price, outcome, closed_at=candle_ts)

    def restore_from_store(self, instrument: str, df: pd.DataFrame,
                           cost_pct_fn: Callable[[float], float],
                           interval_hours: float = 1.0,
                           eod_close: bool = True) -> dict:
        """Rebuild the state lost by a restart from the signals whose outcome
        is still NULL, replaying the candles with the v3 rules:

          - every unresolved signal except the newest is replayed up to the
            next signal; if nothing hit, it is closed as SIGNAL_REVERSED at
            the next signal's entry (what the live engine would have done);
          - the newest unresolved signal becomes the open position and is
            replayed against the candles fetched so far.

        `cost_pct_fn(entry_price)` -> round-trip cost in % for the instrument.
        Returns {"resolved": n, "open": bool}.
        """
        if self._store is None:
            return {"resolved": 0, "open": False}
        rows = self._store.fetch_open(instrument)
        resolved = 0
        for i, row in enumerate(rows):
            ts_utc = _as_utc(row["ts_utc"])
            pos = TrackedPosition(
                instrument=instrument, direction=row["direction"],
                entry_price=row["entry_price"], stop_loss=row["stop_loss"],
                take_profit=row["take_profit"],
                strategy_name=row["strategy"] or "restored",
                eod_close=eod_close, signal_id=row["id"],
                cost_pct=cost_pct_fn(row["entry_price"]),
                signal_candle_ts=(floor_to_interval(ts_utc, interval_hours)
                                  - timedelta(hours=interval_hours)),
            )
            pos.opened_at = ts_utc
            is_newest = (i == len(rows) - 1)
            until = None if is_newest else _as_utc(rows[i + 1]["ts_utc"])
            hit = self._replay(pos, df, until_ts=until)
            if hit is not None:
                exit_price, outcome, candle_ts = hit
                self._close(pos, exit_price, outcome, closed_at=candle_ts)
                resolved += 1
            elif not is_newest:
                nxt = rows[i + 1]
                self._close(pos, nxt["entry_price"], "SIGNAL_REVERSED",
                            closed_at=until)
                resolved += 1
            else:
                self._positions[instrument] = pos
                logger.info(f"Restored open position: {pos.direction} "
                            f"{instrument} @ {pos.entry_price} (signal {pos.signal_id})")
        if resolved:
            logger.info(f"Restored {resolved} pending outcome(s) for {instrument}")
        return {"resolved": resolved, "open": self.has_position(instrument)}

    # ------------------------------------------------------------------
    # Tick-based checks (legacy "vote" mode)
    # ------------------------------------------------------------------

    def check_sl_tp(self, instrument: str, current_price: float,
                    atr: float = 0) -> Optional[TrackedPosition]:
        """Check SL/TP against the live price. Also updates trailing SL."""
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
        now = _utcnow()
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
