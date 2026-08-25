import os
from dataclasses import dataclass, field
from typing import Optional

from dotenv import load_dotenv

load_dotenv()


@dataclass
class InstrumentConfig:
    """Configuration for a single tradeable instrument.

    Cost model:
      - Default: absolute spread = SPREAD_PIPS * PIP_VALUE (price units).
      - If SPREAD_PCT is set (> 0) it takes priority and is interpreted as the
        ROUND-TRIP spread cost as a fraction of price (e.g. 0.003 = 0.30%).
        This is the natural model for crypto CFDs where spread scales with price.

    Per-instrument strategy parameters (SL_ATR .. MIN_RR) override the global
    STRATEGY/AI defaults when set; they are populated by the optimizer.
    """
    SYMBOL: str              # yfinance ticker (for candles/training)
    SYMBOL_DISPLAY: str      # Human-readable name
    MODEL_PATH: str          # Path to trained XGBoost model
    TWELVEDATA_SYMBOL: str = ""  # Twelve Data symbol for real-time prices
    TV_SYMBOL: str = ""      # TradingView symbol for real-time spot prices
    TV_EXCHANGE: str = "cfd"  # TradingView scan endpoint (cfd, forex or crypto)
    CANDLE_INTERVAL: str = "5m"
    HISTORY_PERIOD: str = "5d"
    TRAIN_PERIOD: str = "2y"
    TRAIN_INTERVAL: str = "1h"
    # Period used by the live monitor to fetch TRAIN_INTERVAL candles for the
    # AI path (must cover enough candles for features: >= 60 x 1h).
    AI_HISTORY_PERIOD: str = "30d"
    PRICE_CHANGE_THRESHOLD: float = 0.005
    # Label threshold grid searched by the optimizer. Empty = only
    # PRICE_CHANGE_THRESHOLD is used.
    THRESHOLD_GRID: list = field(default_factory=list)
    SPREAD_PIPS: float = 3.0   # Typical spread in pips (for cost modeling)
    PIP_VALUE: float = 0.01    # Value of 1 pip in price units
    SPREAD_PCT: float = 0.0    # Optional ROUND-TRIP spread as fraction of price
    # Session handling: gold/FX use the global session window; 24/7 markets
    # (crypto) set SESSION_24_7=True and skip the EOD forced close.
    SESSION_24_7: bool = False
    ENABLED: bool = True
    # --- Per-instrument strategy overrides (None -> global defaults) ---
    SL_ATR: Optional[float] = None          # Stop-loss distance in ATR multiples
    TP_ATR: Optional[float] = None          # Take-profit distance in ATR multiples
    CONFIDENCE: Optional[float] = None      # Min AI confidence to act
    ADX_MIN: Optional[float] = None         # Min ADX (trend gate) to trade
    MIN_RR: Optional[float] = None          # Min reward:risk ratio

    # Resolved accessors (fall back to global STRATEGY/AI config)
    def sl_atr(self) -> float:
        return self.SL_ATR if self.SL_ATR is not None else STRATEGY.SCALP_ATR_SL

    def tp_atr(self) -> float:
        return self.TP_ATR if self.TP_ATR is not None else STRATEGY.SCALP_ATR_TP

    def confidence_threshold(self) -> float:
        return self.CONFIDENCE if self.CONFIDENCE is not None else AI.CONFIDENCE_THRESHOLD

    def adx_min(self) -> float:
        return self.ADX_MIN if self.ADX_MIN is not None else STRATEGY.REGIME_ADX_THRESHOLD

    def min_rr(self) -> float:
        return self.MIN_RR if self.MIN_RR is not None else 1.5

    def threshold_grid(self) -> list:
        return self.THRESHOLD_GRID or [self.PRICE_CHANGE_THRESHOLD]


# Twelve Data API key (free tier: 800 req/day, 8/min)
TWELVEDATA_API_KEY = os.getenv("TWELVEDATA_API_KEY", "")

# All monitored instruments
INSTRUMENTS = [
    InstrumentConfig(
        SYMBOL="GC=F",
        SYMBOL_DISPLAY="XAU/USD (Gold)",
        MODEL_PATH="models/gold_xgb.pkl",
        TWELVEDATA_SYMBOL="XAU/USD",
        TV_SYMBOL="OANDA:XAUUSD",
        TV_EXCHANGE="cfd",
        PRICE_CHANGE_THRESHOLD=0.005,
        SPREAD_PIPS=3.0,     # XTB Gold spread ~3 pips ($0.30)
        PIP_VALUE=0.10,      # Gold: 1 pip = $0.10
    ),
    InstrumentConfig(
        SYMBOL="EURUSD=X",
        SYMBOL_DISPLAY="EUR/USD",
        MODEL_PATH="models/eurusd_xgb.pkl",
        TWELVEDATA_SYMBOL="EUR/USD",
        TV_SYMBOL="FX:EURUSD",
        TV_EXCHANGE="forex",
        PRICE_CHANGE_THRESHOLD=0.002,
        SPREAD_PIPS=1.2,     # XTB EUR/USD spread ~1.2 pips
        PIP_VALUE=0.0001,    # Forex: 1 pip = 0.0001
        ENABLED=False,       # Disabled: optimizer found no edge (+0.41%, Sharpe 0.14)
    ),
    InstrumentConfig(
        SYMBOL="GBPUSD=X",
        SYMBOL_DISPLAY="GBP/USD",
        MODEL_PATH="models/gbpusd_xgb.pkl",
        TWELVEDATA_SYMBOL="GBP/USD",
        TV_SYMBOL="FX:GBPUSD",
        TV_EXCHANGE="forex",
        PRICE_CHANGE_THRESHOLD=0.002,
        SPREAD_PIPS=1.5,
        PIP_VALUE=0.0001,
        ENABLED=False,
    ),
]


@dataclass
class MonitorConfig:
    FETCH_INTERVAL_SEC: int = 10  # 10 sec (TradingView is free)
    ANALYSIS_INTERVAL_SEC: int = 60
    CANDLE_LOOKBACK: int = 100
    # AI candles (TRAIN_INTERVAL timeframe) are refreshed at most this often;
    # a new 1h candle only appears hourly so 5 min is plenty.
    AI_CANDLE_REFRESH_SEC: int = 300


@dataclass
class DiscordConfig:
    WEBHOOK_URL: str = os.getenv("DISCORD_WEBHOOK_URL", "")
    BOT_NAME: str = "Trading Monitor"
    # User mention prepended to notifications, e.g. "<@123456789012345678>".
    # Personal - lives only in .env (DISCORD_MENTION), never hardcoded here.
    MENTION_TAG: str = os.getenv("DISCORD_MENTION", "")
    NOTIFY_ON_SIGNAL_CHANGE: bool = True
    SEND_HOURLY_STATUS: bool = False


@dataclass
class AIConfig:
    CONFIDENCE_THRESHOLD: float = 0.60  # Optimized: 0.60 for Gold
    PREDICTION_HORIZON: int = 6
    PRICE_CHANGE_THRESHOLD: float = 0.005
    N_ESTIMATORS: int = 150
    MAX_DEPTH: int = 4
    LEARNING_RATE: float = 0.05
    FEATURE_NAMES: list = None

    def __post_init__(self):
        if self.FEATURE_NAMES is None:
            # Lazy import to avoid circular dependency
            try:
                from ai.feature_engineer import FeatureEngineer
                self.FEATURE_NAMES = FeatureEngineer.FEATURE_NAMES
            except ImportError:
                self.FEATURE_NAMES = []


@dataclass
class StrategyConfig:
    # Scalping (5 min)
    SCALP_RSI_PERIOD: int = 14
    SCALP_RSI_OVERSOLD: float = 30.0
    SCALP_RSI_OVERBOUGHT: float = 70.0
    SCALP_BB_PERIOD: int = 20
    SCALP_BB_STD: float = 2.0
    SCALP_ATR_SL: float = 1.5  # Optimized: 1.5x ATR stop loss
    SCALP_ATR_TP: float = 3.0  # Optimized: 3.0x ATR take profit

    # Momentum (15 min)
    MOM_EMA_FAST: int = 9
    MOM_EMA_SLOW: int = 21
    MOM_MACD_FAST: int = 12
    MOM_MACD_SLOW: int = 26
    MOM_MACD_SIGNAL: int = 9
    MOM_ATR_SL: float = 1.5   # Optimized: aligned with SL=1.5
    MOM_ATR_TP: float = 3.0   # Optimized: aligned with TP=3.0

    # Voting weights
    AI_WEIGHT: float = 0.50
    SCALP_WEIGHT: float = 0.25
    MOMENTUM_WEIGHT: float = 0.25
    VOTE_THRESHOLD: float = 0.35

    # Session filter (UTC hours) - London+NY overlap
    SESSION_START_UTC: int = 8    # 08:00 UTC (London open)
    SESSION_END_UTC: int = 20     # 20:00 UTC (NY close)

    # Auto-close positions before end of day
    EOD_CLOSE_UTC: int = 21       # Force close at 21:00 UTC

    # Trailing stop loss
    TRAILING_SL_ENABLED: bool = True
    TRAILING_SL_ACTIVATION: float = 0.3  # Activate after 0.3% profit
    TRAILING_SL_DISTANCE: float = 1.0    # Trail at 1x ATR distance

    # Regime filter - Optimized: ADX >= 25 required to trade
    REGIME_ADX_THRESHOLD: float = 25.0   # ADX < 25 = no new trades
    REGIME_ADX_TRENDING: float = 25.0    # ADX >= 25 = trending (OK to trade)


@dataclass
class CostConfig:
    """Transaction cost modeling for realistic backtesting."""
    COMMISSION_PCT: float = 0.0    # XTB CFDs: no commission (spread only)
    SLIPPAGE_MULTIPLIER: float = 0.5  # Slippage = 0.5x spread on fast markets

    def round_trip_cost_pct(self, instrument: InstrumentConfig, price: float) -> float:
        """Total ROUND-TRIP cost as a fraction of price (spread + slippage).

        Priority: SPREAD_PCT (already round-trip) if set, otherwise the
        pip-based absolute spread model (one-way spread doubled).
        """
        if instrument.SPREAD_PCT > 0:
            base = instrument.SPREAD_PCT
        else:
            if price <= 0:
                return 0.0
            base = (instrument.SPREAD_PIPS * instrument.PIP_VALUE) / price * 2
        return base * (1 + self.SLIPPAGE_MULTIPLIER) + self.COMMISSION_PCT * 2

    def get_total_cost_pct(self, instrument: InstrumentConfig, price: float) -> float:
        """Backwards-compatible alias for round_trip_cost_pct."""
        return self.round_trip_cost_pct(instrument, price)


# Global instances
MONITOR = MonitorConfig()
DISCORD = DiscordConfig()
AI = AIConfig()
STRATEGY = StrategyConfig()
COSTS = CostConfig()
