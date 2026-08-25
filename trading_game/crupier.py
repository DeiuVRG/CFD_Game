"""The Crupier: central authority of the game (spec Part 1.3).

Provides data & indicators, validates and executes ALL trades with realistic
costs and liquidity constraints, manages portfolios, keeps the full audit
log, enforces temporal integrity (players only ever see data up to the
current simulation date), approves/registers external data sources, applies
violation & elimination rules and validates recalibrations.
"""
import hashlib
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from trading_game.config import GameConfig
from trading_game.costs import LiquidityConstraints, RealisticCosts
from trading_game.indicators import CATALOG, compute_signals

logger = logging.getLogger(__name__)


@dataclass
class Position:
    quantity: float = 0.0
    avg_cost: float = 0.0


@dataclass
class Portfolio:
    cash: float
    positions: Dict[str, Position] = field(default_factory=dict)

    def position_qty(self, symbol: str) -> float:
        pos = self.positions.get(symbol)
        return pos.quantity if pos else 0.0

    def open_symbols(self) -> List[str]:
        return [s for s, p in self.positions.items() if p.quantity > 0]


class Crupier:
    def __init__(self, config: GameConfig, data: Dict[str, pd.DataFrame],
                 market: pd.DataFrame, vix: pd.DataFrame):
        self.config = config
        self.costs = RealisticCosts(config)
        self.liquidity = LiquidityConstraints(config)

        # --- Precomputed per-symbol series (indexed by date) ---
        self._close: Dict[str, pd.Series] = {}
        self._adv: Dict[str, pd.Series] = {}
        self._signals: Dict[str, pd.DataFrame] = {}
        self._raw: Dict[str, pd.DataFrame] = {}
        for symbol, df in data.items():
            idx = df.set_index("timestamp")
            self._close[symbol] = idx["close"]
            self._adv[symbol] = idx["volume"].rolling(20).mean().bfill()
            sig = compute_signals(df)
            sig.index = df["timestamp"]
            self._signals[symbol] = sig
            self._raw[symbol] = idx

        self.market = market
        self.vix = vix
        self._market_close = market.set_index("timestamp")["close"]

        # --- Player state ---
        self.portfolios: Dict[str, Portfolio] = {}
        self.methods: Dict[str, str] = {}
        self.indicator_requests: Dict[str, List[str]] = {}
        self.trade_log: List[dict] = []
        self.closed_trades: Dict[str, List[dict]] = {}
        self.monthly_trade_counts: Dict[str, Dict[str, int]] = {}
        self.daily_equity: Dict[str, List[tuple]] = {}
        self.external_sources: List[dict] = []
        self.external_data_registry: List[dict] = []
        self.eliminated: Dict[str, str] = {}

        self.current_date: Optional[pd.Timestamp] = None

    # ------------------------------------------------------------------
    # Registration & data service
    # ------------------------------------------------------------------

    def register_player(self, player_id: str, indicators: List[str],
                        method_description: str = "") -> dict:
        unknown = [i for i in indicators if i not in CATALOG]
        if unknown:
            return {"status": "REJECTED", "reason": f"unknown indicators {unknown}"}
        if not (self.config.min_indicators <= len(indicators) <= self.config.max_indicators):
            return {"status": "REJECTED",
                    "reason": f"need {self.config.min_indicators}-"
                              f"{self.config.max_indicators} indicators"}
        self.portfolios[player_id] = Portfolio(cash=self.config.initial_capital)
        self.indicator_requests[player_id] = list(indicators)
        self.methods[player_id] = method_description
        self.closed_trades[player_id] = []
        self.monthly_trade_counts[player_id] = {}
        self.daily_equity[player_id] = []
        return {"status": "APPROVED"}

    def trading_days(self, start: str, end: str) -> pd.DatetimeIndex:
        s = self._market_close.loc[pd.Timestamp(start):pd.Timestamp(end)]
        return s.index

    def set_current_date(self, date: pd.Timestamp):
        self.current_date = pd.Timestamp(date)

    def provide_indicators(self, player_id: str, symbol: str,
                           until: pd.Timestamp = None) -> pd.DataFrame:
        """Indicator signals for one symbol UP TO `until` (temporal
        integrity: never returns future rows). Only the player's requested
        indicators are served."""
        until = pd.Timestamp(until) if until is not None else self.current_date
        cols = self.indicator_requests.get(player_id, CATALOG)
        sig = self._signals.get(symbol)
        if sig is None:
            return pd.DataFrame()
        return sig.loc[:until, [c for c in cols if c in sig.columns]]

    def provide_ohlcv(self, symbol: str, until: pd.Timestamp = None) -> pd.DataFrame:
        until = pd.Timestamp(until) if until is not None else self.current_date
        raw = self._raw.get(symbol)
        if raw is None:
            return pd.DataFrame()
        return raw.loc[:until]

    def price(self, symbol: str, date: pd.Timestamp = None) -> Optional[float]:
        date = pd.Timestamp(date) if date is not None else self.current_date
        series = self._close.get(symbol)
        if series is None:
            return None
        value = series.asof(date)
        return None if pd.isna(value) else float(value)

    def adv(self, symbol: str, date: pd.Timestamp = None) -> float:
        date = pd.Timestamp(date) if date is not None else self.current_date
        series = self._adv.get(symbol)
        if series is None:
            return 0.0
        value = series.asof(date)
        return 0.0 if pd.isna(value) else float(value)

    # ------------------------------------------------------------------
    # External data (spec 1.2): approval + transparent registry
    # ------------------------------------------------------------------

    def approve_external_source(self, player_id: str, request: dict) -> dict:
        allowed_types = {"news", "financial_reports", "sentiment",
                         "economic", "other"}
        per_player = [s for s in self.external_sources
                      if s["player_id"] == player_id]
        if len(per_player) >= 10:
            return {"status": "REJECTED", "reason": "max 10 sources per player"}
        if request.get("tip_sursa") not in allowed_types:
            return {"status": "REJECTED", "reason": "unknown source type"}
        source_id = f"src_{len(self.external_sources) + 1}"
        record = {"source_id": source_id, "player_id": player_id, **request}
        self.external_sources.append(record)
        return {"status": "APPROVED", "source_id": source_id}

    def register_external_data(self, player_id: str, source_id: str,
                               data, as_of: pd.Timestamp) -> dict:
        """Registers procured data with a verification hash. as_of documents
        the real-world availability date (no look-ahead)."""
        as_of = pd.Timestamp(as_of)
        if self.current_date is not None and as_of > self.current_date:
            return {"status": "REJECTED", "reason": "look-ahead bias: data from the future"}
        digest = hashlib.sha256(str(data).encode()).hexdigest()
        receipt = {
            "player_id": player_id, "source_id": source_id,
            "as_of": str(as_of), "sha256": digest,
            "bytes": len(str(data)),
        }
        self.external_data_registry.append(receipt)
        return {"status": "REGISTERED", **receipt}

    # ------------------------------------------------------------------
    # Trading (spec 3.2): every order goes through here
    # ------------------------------------------------------------------

    def _month_key(self, date: pd.Timestamp) -> str:
        return f"{date.year}-{date.month:02d}"

    def execute_trade(self, player_id: str, order: dict) -> dict:
        """order = {action: BUY|SELL, company: str, quantity: float}"""
        if player_id in self.eliminated:
            return {"status": "REJECTED", "reason": "player eliminated"}
        portfolio = self.portfolios.get(player_id)
        if portfolio is None:
            return {"status": "REJECTED", "reason": "unknown player"}

        date = self.current_date
        action = order.get("action")
        symbol = order.get("company")
        qty = float(order.get("quantity", 0))

        if action not in ("BUY", "SELL") or qty <= 0:
            return {"status": "REJECTED", "reason": "invalid order"}
        price = self.price(symbol, date)
        if price is None or price <= 0:
            return {"status": "REJECTED", "reason": f"no price for {symbol}"}

        month = self._month_key(date)
        counts = self.monthly_trade_counts[player_id]
        if counts.get(month, 0) >= self.config.max_trades_per_month:
            return {"status": "REJECTED",
                    "reason": f"max {self.config.max_trades_per_month} trades/month"}

        adv = self.adv(symbol, date)

        if action == "SELL":
            held = portfolio.position_qty(symbol)
            if held <= 0:
                return {"status": "REJECTED", "reason": "no position (long-only game)"}
            qty = min(qty, held)

        execution = self.liquidity.limit_order(qty, price, adv)
        exec_qty = execution.executed_qty
        exec_price = execution.execution_price
        value = exec_qty * exec_price
        cost = self.costs.transaction_cost(value, exec_qty, adv)

        if action == "BUY":
            if portfolio.position_qty(symbol) == 0 and \
                    len(portfolio.open_symbols()) >= self.config.max_positions:
                return {"status": "REJECTED",
                        "reason": f"max {self.config.max_positions} positions"}
            total = value + cost
            if total > portfolio.cash:
                # Scale down to affordable size instead of hard-rejecting
                affordable = portfolio.cash / (exec_price * (1 + 0.01)) - 1
                if affordable < 1:
                    return {"status": "REJECTED", "reason": "insufficient cash"}
                exec_qty = float(int(affordable))
                value = exec_qty * exec_price
                cost = self.costs.transaction_cost(value, exec_qty, adv)
                total = value + cost
                if total > portfolio.cash or exec_qty <= 0:
                    return {"status": "REJECTED", "reason": "insufficient cash"}
            pos = portfolio.positions.setdefault(symbol, Position())
            new_qty = pos.quantity + exec_qty
            pos.avg_cost = (pos.avg_cost * pos.quantity + value) / new_qty
            pos.quantity = new_qty
            portfolio.cash -= total
        else:  # SELL
            pos = portfolio.positions[symbol]
            profit = (exec_price - pos.avg_cost) * exec_qty - cost
            self.closed_trades[player_id].append({
                "date": str(date.date()), "symbol": symbol,
                "quantity": exec_qty, "entry": pos.avg_cost,
                "exit": exec_price, "cost": cost, "profit": profit,
            })
            pos.quantity -= exec_qty
            if pos.quantity <= 1e-9:
                del portfolio.positions[symbol]
            portfolio.cash += value - cost

        counts[month] = counts.get(month, 0) + 1

        record = {
            "date": str(date.date()), "player_id": player_id,
            "action": action, "symbol": symbol,
            "requested_qty": qty, "executed_qty": exec_qty,
            "price": round(exec_price, 4), "cost": round(cost, 2),
            "note": execution.note, "cash_after": round(portfolio.cash, 2),
        }
        self.trade_log.append(record)

        return {
            "status": "EXECUTED",
            "executed_qty": exec_qty,
            "execution_price": exec_price,
            "total_cost": cost,
            "new_cash": portfolio.cash,
            "note": execution.note,
        }

    # ------------------------------------------------------------------
    # Valuation & bookkeeping
    # ------------------------------------------------------------------

    def equity(self, player_id: str, date: pd.Timestamp = None) -> float:
        portfolio = self.portfolios[player_id]
        date = pd.Timestamp(date) if date is not None else self.current_date
        total = portfolio.cash
        for symbol, pos in portfolio.positions.items():
            price = self.price(symbol, date)
            if price is not None:
                total += pos.quantity * price
        return float(total)

    def snapshot_equity(self, active_players: List[str], date: pd.Timestamp):
        for pid in active_players:
            self.daily_equity[pid].append((date, self.equity(pid, date)))

    def equity_series(self, player_id: str) -> pd.Series:
        snaps = self.daily_equity[player_id]
        if not snaps:
            return pd.Series(dtype=float)
        dates, values = zip(*snaps)
        return pd.Series(values, index=pd.DatetimeIndex(dates))

    def trades_in_month(self, player_id: str, month_key: str) -> int:
        return self.monthly_trade_counts[player_id].get(month_key, 0)

    # ------------------------------------------------------------------
    # Rules: violations, elimination, recalibration (spec Parts III & V)
    # ------------------------------------------------------------------

    def check_violations(self, player_id: str, month_key: str) -> Optional[str]:
        n = self.trades_in_month(player_id, month_key)
        if n < self.config.min_trades_per_month:
            return "no mandatory monthly trade"
        if n > self.config.max_trades_per_month:
            return f"overtrading (> {self.config.max_trades_per_month}/month)"
        equity = self.equity(player_id)
        if equity < self.config.initial_capital * self.config.catastrophic_drawdown_equity:
            return "drawdown > 50%"
        return None

    def evaluate_elimination(self, scores: Dict[str, float]) -> Optional[str]:
        """Monthly rule: eliminate the last player when they trail the
        second-last by >= elimination_gap (relative)."""
        if len(scores) <= 2:
            return None
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        last_id, last_score = ranked[-1]
        _, second_last_score = ranked[-2]
        if second_last_score <= 0:
            return None
        distance = (second_last_score - last_score) / second_last_score
        if distance >= self.config.elimination_gap:
            return last_id
        return None

    def eliminate(self, player_id: str, reason: str):
        self.eliminated[player_id] = reason
        logger.info(f"ELIMINATED {player_id}: {reason}")

    def validate_recalibration(self, old_weights: Dict[str, float],
                               new_weights: Dict[str, float]) -> bool:
        """Reject drastic shifts: max 30% absolute change per indicator, and
        the new weights must be a valid normalized set."""
        if not new_weights:
            return False
        total = sum(new_weights.values())
        if abs(total - 1.0) > 1e-6:
            return False
        keys = set(old_weights) | set(new_weights)
        for key in keys:
            shift = abs(new_weights.get(key, 0.0) - old_weights.get(key, 0.0))
            if shift > self.config.max_weight_shift + 1e-9:
                return False
        return True
