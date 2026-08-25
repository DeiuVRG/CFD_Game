"""Base player: weighted-indicator decision logic + recalibration blending.

Each player:
  - requests a set of indicators from the catalog (3..20),
  - determines WEIGHTS (sum=1, each in [0,1]) with its own method on the
    TRAINING segment,
  - may keep an ORIENTATION (+1/-1) per indicator - its interpretation of
    the standardized signal (e.g. momentum players flip oscillators from
    contrarian to trend-following). Freedom of method is total (spec 1.1);
    weights themselves stay non-negative and normalized (spec 2.1 step 4).
  - decides monthly trades from score = sum(w_i * orient_i * signal_i),
    BUY above +0.3, SELL below -0.3 (long-only interpretation),
  - respects the mandatory min-1-trade-per-month rule with a fallback trade.
"""
from typing import Dict, List

import numpy as np
import pandas as pd

from trading_game.config import GameConfig


class BasePlayer:
    name = "base"
    method_description = ""

    def __init__(self, player_id: str, config: GameConfig, seed: int = 0):
        self.id = player_id
        self.config = config
        self.seed = seed
        self.indicators: List[str] = []
        self.weights: Dict[str, float] = {}
        self.orientation: Dict[str, float] = {}
        self.recalibrations = 0

    # ------------------------------------------------------------------
    # Weight determination (per-player method)
    # ------------------------------------------------------------------

    def fit(self, train_data: Dict[str, dict]):
        """train_data: {symbol: {"signals": DataFrame, "close": Series}}
        restricted to the training segment. Must set self.weights (and
        optionally self.orientation)."""
        raise NotImplementedError

    def _normalize(self, raw: Dict[str, float]) -> Dict[str, float]:
        clipped = {k: max(0.0, float(v)) for k, v in raw.items()}
        total = sum(clipped.values())
        if total <= 0:
            n = len(self.indicators)
            return {k: 1.0 / n for k in self.indicators}
        return {k: v / total for k, v in clipped.items()}

    def get_orientation(self, indicator: str) -> float:
        return self.orientation.get(indicator, 1.0)

    # ------------------------------------------------------------------
    # Monthly decisions
    # ------------------------------------------------------------------

    def score_symbol(self, signals_row: pd.Series) -> float:
        score = 0.0
        for indicator, weight in self.weights.items():
            value = signals_row.get(indicator)
            if value is None or pd.isna(value):
                continue
            score += weight * self.get_orientation(indicator) * float(value)
        return score

    def decide_trades(self, crupier, date: pd.Timestamp) -> List[dict]:
        cfg = self.config
        scores: Dict[str, float] = {}
        for symbol in cfg.universe:
            sig = crupier.provide_indicators(self.id, symbol, until=date)
            if sig.empty or sig.iloc[-1].isna().all():
                continue
            scores[symbol] = self.score_symbol(sig.iloc[-1])

        portfolio = crupier.portfolios[self.id]
        held = portfolio.open_symbols()
        equity = crupier.equity(self.id, date)
        orders: List[dict] = []

        # SELL positions whose score turned bearish
        sells = []
        for symbol in held:
            if scores.get(symbol, 0.0) < cfg.signal_sell_threshold:
                qty = portfolio.position_qty(symbol)
                if qty > 0:
                    orders.append({"action": "SELL", "company": symbol,
                                   "quantity": qty})
                    sells.append(symbol)

        # BUY the strongest bullish candidates into free slots
        candidates = sorted(
            ((s, v) for s, v in scores.items()
             if v > cfg.signal_buy_threshold and s not in held),
            key=lambda kv: kv[1], reverse=True,
        )
        free_slots = cfg.max_positions - (len(held) - len(sells))
        for symbol, score in candidates[:max(0, min(free_slots, 5))]:
            price = crupier.price(symbol, date)
            if not price:
                continue
            target_value = equity * 0.12 * min(1.0, abs(score))
            qty = int(min(target_value, portfolio.cash * 0.9) / price)
            if qty >= 1:
                orders.append({"action": "BUY", "company": symbol,
                               "quantity": qty})

        # Mandatory monthly trade fallback: trade 1 share of the best-scored
        # symbol (elimination for inactivity is worse than a tiny trade).
        if not orders and scores:
            best_symbol = max(scores, key=scores.get)
            price = crupier.price(best_symbol, date)
            if price and portfolio.cash > price * 1.02:
                orders.append({"action": "BUY", "company": best_symbol,
                               "quantity": 1})
            elif held:
                orders.append({"action": "SELL", "company": held[0],
                               "quantity": max(1.0, portfolio.position_qty(held[0]) * 0.1)})

        return orders

    # ------------------------------------------------------------------
    # Quarterly recalibration (spec 3.4): refit, then blend toward the old
    # weights so no indicator shifts more than max_weight_shift.
    # ------------------------------------------------------------------

    def propose_recalibration(self, data_until_now: Dict[str, dict]) -> Dict[str, float]:
        old = dict(self.weights)
        try:
            self.fit(data_until_now)
        except Exception:
            self.weights = old
            return old
        fresh = dict(self.weights)
        self.weights = old  # the crupier decides whether the shift is legal

        diffs = {k: fresh.get(k, 0.0) - old.get(k, 0.0)
                 for k in set(old) | set(fresh)}
        max_diff = max((abs(d) for d in diffs.values()), default=0.0)
        if max_diff <= self.config.max_weight_shift:
            return fresh
        alpha = self.config.max_weight_shift / max_diff
        blended = {k: old.get(k, 0.0) + alpha * d for k, d in diffs.items()}
        total = sum(blended.values())
        return {k: v / total for k, v in blended.items()} if total > 0 else old

    def apply_recalibration(self, new_weights: Dict[str, float]):
        self.weights = dict(new_weights)
        self.recalibrations += 1

    # ------------------------------------------------------------------

    @staticmethod
    def forward_returns(close: pd.Series, horizon: int = 21) -> pd.Series:
        return close.shift(-horizon) / close - 1
