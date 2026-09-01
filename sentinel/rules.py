"""Hard rules - pure functions, enforced in code regardless of the model.

The model may only: approve/veto a deterministic signal, scale the size
DOWN (size_fraction <= 1), close a position early, or tighten its stop.
Everything else is rejected here.
"""
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Optional, Tuple

from sentinel.config import SentinelConfig
from sentinel.schema import ManageDecision, OpenDecision


@dataclass
class RiskState:
    equity: float = 0.0
    day_start_equity: float = 0.0
    trades_today: int = 0
    open_positions: List[dict] = field(default_factory=list)   # broker positions

    @property
    def daily_loss_pct(self) -> float:
        if self.day_start_equity <= 0:
            return 0.0
        return max(0.0, (self.day_start_equity - self.equity) / self.day_start_equity)


def parse_ts(ts_utc: str) -> datetime:
    t = datetime.fromisoformat(ts_utc.replace("Z", "+00:00"))
    return t if t.tzinfo else t.replace(tzinfo=timezone.utc)


def signal_age_sec(signal_row: dict, now: datetime) -> float:
    return (now - parse_ts(signal_row["ts_utc"])).total_seconds()


def risk_reward(entry: float, sl: float, tp: float) -> float:
    risk = abs(entry - sl)
    return abs(tp - entry) / risk if risk > 0 else 0.0


def check_open(decision: Optional[OpenDecision], signal_row: dict, now: datetime,
               cfg: SentinelConfig, state: RiskState, epic: str) -> Tuple[bool, str]:
    """(approved, reason). The reason names the FIRST rule that blocked."""
    if decision is None:
        return False, "NO_DECISION"          # fail closed
    if decision.action != "APPROVE":
        return False, "VETO"
    if decision.size_fraction <= 0:
        return False, "ZERO_SIZE"
    if signal_age_sec(signal_row, now) > cfg.signal_max_age_sec:
        return False, "STALE"
    if state.daily_loss_pct >= cfg.max_daily_loss:
        return False, "LIMIT_DAILY_LOSS"
    if state.trades_today >= cfg.max_trades_per_day:
        return False, "LIMIT_TRADES"
    if len(state.open_positions) >= cfg.max_concurrent_positions:
        return False, "LIMIT_POSITIONS"
    if any(p.get("epic") == epic for p in state.open_positions):
        return False, "DUPLICATE_EPIC"
    entry, sl, tp = signal_row["entry_price"], signal_row["stop_loss"], signal_row["take_profit"]
    if not entry or not sl or sl == entry:
        return False, "INVALID_SL"
    if risk_reward(entry, sl, tp or entry) < cfg.min_risk_reward:
        return False, "LOW_RR"
    return True, "OK"


def check_manage(decision: Optional[ManageDecision], position: dict,
                 current_price: float) -> Tuple[str, Optional[float], str]:
    """-> (action, new_stop_loss, reason). Only HOLD / CLOSE / TIGHTEN_SL;
    a stop may only move in the position's favour and never past the
    current price."""
    if decision is None:
        return "HOLD", None, "NO_DECISION"
    if decision.action == "CLOSE":
        return "CLOSE", None, "OK"
    if decision.action == "TIGHTEN_SL":
        new_sl = decision.new_stop_loss
        cur_sl = position.get("stop_level")
        if new_sl is None or current_price <= 0:
            return "HOLD", None, "INVALID_SL"
        if position["direction"] == "BUY":
            ok = new_sl < current_price and (cur_sl is None or new_sl > cur_sl)
        else:
            ok = new_sl > current_price and (cur_sl is None or new_sl < cur_sl)
        return ("TIGHTEN_SL", new_sl, "OK") if ok else ("HOLD", None, "INVALID_SL")
    return "HOLD", None, "OK"


def position_size(equity: float, risk_pct: float, entry: float, stop_loss: float,
                  min_size: float, max_size: float) -> float:
    """Size so that hitting the SL loses `risk_pct` of equity (quote currency
    = account currency; same formula as execution_capital's PositionSizer)."""
    sl_distance = abs(entry - stop_loss)
    if equity <= 0 or risk_pct <= 0 or sl_distance <= 0:
        return 0.0
    size = min(equity * risk_pct / sl_distance, max_size)
    if size < min_size:
        return 0.0
    return round(size, 2)
