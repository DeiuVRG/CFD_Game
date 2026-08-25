from dataclasses import dataclass
from datetime import datetime


@dataclass
class Signal:
    epic: str
    direction: str          # "BUY" or "SELL"
    entry_price: float
    stop_loss: float
    take_profit: float
    strategy_name: str
    strength: float         # 0.0 to 1.0
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()

    @property
    def risk_reward_ratio(self) -> float:
        risk = abs(self.entry_price - self.stop_loss)
        reward = abs(self.take_profit - self.entry_price)
        if risk == 0:
            return 0.0
        return reward / risk


@dataclass
class ApprovedTrade:
    signal: Signal
    size: float
    risk_amount: float
