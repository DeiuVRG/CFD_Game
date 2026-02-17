import os
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()


@dataclass
class InstrumentConfig:
    """Configuration for a single tradeable instrument."""
    SYMBOL: str              # yfinance ticker (for candles/training)
    SYMBOL_DISPLAY: str      # Human-readable name
    MODEL_PATH: str          # Path to trained XGBoost model
    TWELVEDATA_SYMBOL: str = ""  # Twelve Data symbol for real-time prices
    CANDLE_INTERVAL: str = "5m"
    HISTORY_PERIOD: str = "5d"
    TRAIN_PERIOD: str = "2y"
    TRAIN_INTERVAL: str = "1h"
    PRICE_CHANGE_THRESHOLD: float = 0.005
    ENABLED: bool = True


# Twelve Data API key (free tier: 800 req/day, 8/min)
TWELVEDATA_API_KEY = os.getenv("TWELVEDATA_API_KEY", "")

# All monitored instruments
INSTRUMENTS = [
    InstrumentConfig(
        SYMBOL="GC=F",
        SYMBOL_DISPLAY="XAU/USD (Gold)",
        MODEL_PATH="models/gold_xgb.pkl",
        TWELVEDATA_SYMBOL="XAU/USD",
        PRICE_CHANGE_THRESHOLD=0.005,
    ),
    InstrumentConfig(
        SYMBOL="EURUSD=X",
        SYMBOL_DISPLAY="EUR/USD",
        MODEL_PATH="models/eurusd_xgb.pkl",
        TWELVEDATA_SYMBOL="EUR/USD",
        PRICE_CHANGE_THRESHOLD=0.002,
    ),
    InstrumentConfig(
        SYMBOL="GBPUSD=X",
        SYMBOL_DISPLAY="GBP/USD",
        MODEL_PATH="models/gbpusd_xgb.pkl",
        TWELVEDATA_SYMBOL="GBP/USD",
        PRICE_CHANGE_THRESHOLD=0.002,
        ENABLED=False,  # Dezactivat
    ),
]


@dataclass
class MonitorConfig:
    FETCH_INTERVAL_SEC: int = 60  # 1 min (batch request = 1 call for all instruments)
    ANALYSIS_INTERVAL_SEC: int = 60
    CANDLE_LOOKBACK: int = 100


@dataclass
class DiscordConfig:
    WEBHOOK_URL: str = os.getenv("DISCORD_WEBHOOK_URL", "")
    BOT_NAME: str = "Trading Monitor"
    MENTION_TAG: str = os.getenv("DISCORD_MENTION", "<@266234847532548096>")
    NOTIFY_ON_SIGNAL_CHANGE: bool = True
    SEND_HOURLY_STATUS: bool = False


@dataclass
class AIConfig:
    CONFIDENCE_THRESHOLD: float = 0.55
    PREDICTION_HORIZON: int = 6
    PRICE_CHANGE_THRESHOLD: float = 0.005
    N_ESTIMATORS: int = 200
    MAX_DEPTH: int = 6
    LEARNING_RATE: float = 0.1
    FEATURE_NAMES: list = None

    def __post_init__(self):
        if self.FEATURE_NAMES is None:
            self.FEATURE_NAMES = [
                "rsi_14", "rsi_7", "macd_hist", "macd_signal_dist",
                "bb_position", "bb_width", "atr_14", "ema_ratio",
                "price_change_5", "price_change_10", "price_change_30",
                "high_low_range", "close_vs_open", "volume_change",
                "hour_sin", "hour_cos", "day_of_week",
                "upper_shadow", "lower_shadow", "consecutive_direction",
            ]


@dataclass
class StrategyConfig:
    # Scalping (5 min)
    SCALP_RSI_PERIOD: int = 14
    SCALP_RSI_OVERSOLD: float = 30.0
    SCALP_RSI_OVERBOUGHT: float = 70.0
    SCALP_BB_PERIOD: int = 20
    SCALP_BB_STD: float = 2.0
    SCALP_ATR_SL: float = 1.5

    # Momentum (15 min)
    MOM_EMA_FAST: int = 9
    MOM_EMA_SLOW: int = 21
    MOM_MACD_FAST: int = 12
    MOM_MACD_SLOW: int = 26
    MOM_MACD_SIGNAL: int = 9
    MOM_ATR_SL: float = 2.0
    MOM_ATR_TP: float = 3.0

    # Voting weights
    AI_WEIGHT: float = 0.50
    SCALP_WEIGHT: float = 0.25
    MOMENTUM_WEIGHT: float = 0.25
    VOTE_THRESHOLD: float = 0.35


# Global instances
MONITOR = MonitorConfig()
DISCORD = DiscordConfig()
AI = AIConfig()
STRATEGY = StrategyConfig()
