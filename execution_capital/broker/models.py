from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List


@dataclass
class PriceBar:
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float


@dataclass
class MarketInfo:
    epic: str
    instrument_name: str
    min_deal_size: float
    max_deal_size: float
    pip_size: float
    margin_factor: float
    market_open: bool


@dataclass
class OpenPosition:
    deal_id: str
    epic: str
    direction: str          # "BUY" or "SELL"
    size: float
    open_level: float
    stop_level: Optional[float]
    profit_level: Optional[float]
    profit_loss: float
    created_date: str


@dataclass
class TradeConfirmation:
    deal_id: str
    deal_reference: str
    status: str             # "OPEN", "REJECTED", etc.
    reason: str
    level: float
    size: float
    direction: str
    profit: float
    affected_deals: List[dict] = field(default_factory=list)


@dataclass
class AccountInfo:
    account_id: str
    balance: float
    deposit: float
    profit_loss: float
    available: float
    currency: str
