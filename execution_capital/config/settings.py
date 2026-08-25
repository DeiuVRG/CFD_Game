from dataclasses import dataclass, field
from typing import List


@dataclass
class BrokerConfig:
    DEMO_URL: str = "https://demo-api-capital.backend-capital.com"
    LIVE_URL: str = "https://api-capital.backend-capital.com"
    SESSION_REFRESH_SEC: int = 480  # Refresh at 8 min (timeout is 10 min)
    RATE_LIMIT_SEC: float = 0.12   # ~10 req/sec max
    ORDER_RATE_LIMIT_SEC: float = 0.15
    MAX_RETRIES: int = 3
    RETRY_DELAY_SEC: float = 2.0


@dataclass
class RiskConfig:
    MAX_RISK_PER_TRADE: float = 0.01      # 1% of equity (was 3% - too hot)
    MAX_DAILY_LOSS: float = 0.03           # 3% of day-start equity (was 10%)
    MAX_CONCURRENT_POSITIONS: int = 3
    MAX_TRADES_PER_DAY: int = 5            # Hard cap on new trades per day
    MAX_FX_POSITIONS: int = 1              # Anti-correlation: max 1 FX position
    SL_ATR_MULTIPLIER: float = 1.5
    TP_ATR_MULTIPLIER: float = 2.0
    MIN_RISK_REWARD: float = 1.0           # Minimum 1:1 R:R


@dataclass
class ScalpingConfig:
    TIMEFRAME: str = "MINUTE_5"
    RSI_PERIOD: int = 14
    RSI_OVERSOLD: float = 30.0
    RSI_OVERBOUGHT: float = 70.0
    BB_PERIOD: int = 20
    BB_STD: float = 2.0


@dataclass
class MomentumConfig:
    TIMEFRAME: str = "MINUTE_15"
    EMA_FAST: int = 9
    EMA_SLOW: int = 21
    MACD_FAST: int = 12
    MACD_SLOW: int = 26
    MACD_SIGNAL: int = 9


@dataclass
class MeanReversionConfig:
    TIMEFRAME: str = "MINUTE_5"
    BB_PERIOD: int = 20
    BB_STD: float = 2.0
    RSI_PERIOD: int = 14
    RSI_OVERSOLD: float = 25.0
    RSI_OVERBOUGHT: float = 75.0
    BB_WIDTH_LOOKBACK: int = 50
    TRENDING_MULTIPLIER: float = 1.5


@dataclass
class TradingConfig:
    # Only pairs QUOTED in the account currency (USD). USDJPY was removed:
    # its P&L accrues in JPY and PositionSizer has no quote-currency
    # conversion source, so sizing it would be wrong. Re-add non-USD-quoted
    # pairs only once a conversion rate feed exists (see PositionSizer).
    SYMBOLS: List[str] = field(default_factory=lambda: [
        "EURUSD",
        "GBPUSD",
    ])
    SCAN_INTERVAL_SEC: int = 30
    CANDLE_LOOKBACK: int = 100


# Global instances
BROKER = BrokerConfig()
RISK = RiskConfig()
SCALPING = ScalpingConfig()
MOMENTUM = MomentumConfig()
MEAN_REVERSION = MeanReversionConfig()
TRADING = TradingConfig()
