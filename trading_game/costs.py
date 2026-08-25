"""Realistic transaction costs and liquidity constraints (spec Part III.3)."""
import math
from dataclasses import dataclass

from trading_game.config import GameConfig


@dataclass
class ExecutionResult:
    executed_qty: float
    execution_price: float
    cost: float
    rejected_qty: float = 0.0
    note: str = ""


class RealisticCosts:
    """Commission + spread + size-dependent slippage, per transaction."""

    def __init__(self, config: GameConfig):
        self.config = config

    def transaction_cost(self, value: float, order_qty: float,
                         avg_daily_volume: float) -> float:
        if value <= 0:
            return 0.0
        adv = max(avg_daily_volume, 1.0)
        market_impact = (order_qty / adv) * 0.01
        slippage_total = self.config.slippage_base + market_impact
        variable = value * (
            self.config.commission_percent
            + self.config.spread_percent
            + slippage_total
        )
        return max(variable, self.config.commission_fixed)


class LiquidityConstraints:
    """Max 1% of average daily volume per order; larger orders execute
    partially with a deteriorated price (spec's multi-day pressure model)."""

    def __init__(self, config: GameConfig):
        self.config = config

    def limit_order(self, desired_qty: float, price: float,
                    avg_daily_volume: float) -> ExecutionResult:
        max_qty = max(avg_daily_volume * self.config.max_percent_daily_volume, 1.0)
        if desired_qty <= max_qty:
            return ExecutionResult(executed_qty=desired_qty,
                                   execution_price=price, cost=0.0)
        days_needed = math.ceil(desired_qty / max_qty)
        deteriorated = price * (1 + 0.002 * days_needed)
        return ExecutionResult(
            executed_qty=max_qty,
            execution_price=deteriorated,
            cost=0.0,
            rejected_qty=desired_qty - max_qty,
            note=f"partial fill (liquidity cap, {days_needed}d pressure)",
        )
