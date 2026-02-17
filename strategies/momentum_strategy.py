import logging
from typing import Optional

import pandas as pd

from config.settings import MOMENTUM, RISK
from data.indicators import Indicators
from engine.signal import Signal
from strategies.base_strategy import BaseStrategy

logger = logging.getLogger(__name__)


class MomentumStrategy(BaseStrategy):
    """MACD + EMA crossover momentum strategy on 15-minute candles.

    BUY:  EMA(9) crosses above EMA(21) + MACD histogram positive
    SELL: EMA(9) crosses below EMA(21) + MACD histogram negative
    Target: ATR-based take profit (1.5x risk)
    """

    def __init__(self):
        super().__init__(name="Momentum", timeframe=MOMENTUM.TIMEFRAME)

    def get_required_candles(self) -> int:
        return MOMENTUM.MACD_SLOW + MOMENTUM.MACD_SIGNAL + 10

    def analyze(self, epic: str, df: pd.DataFrame) -> Optional[Signal]:
        if len(df) < self.get_required_candles():
            return None

        close = df["close"]
        high = df["high"]
        low = df["low"]

        ema_fast = Indicators.ema(close, MOMENTUM.EMA_FAST)
        ema_slow = Indicators.ema(close, MOMENTUM.EMA_SLOW)
        _, _, histogram = Indicators.macd(
            close, MOMENTUM.MACD_FAST, MOMENTUM.MACD_SLOW, MOMENTUM.MACD_SIGNAL,
        )
        atr = Indicators.atr(high, low, close)

        price = close.iloc[-1]
        ef = ema_fast.iloc[-1]
        ef_prev = ema_fast.iloc[-2]
        es = ema_slow.iloc[-1]
        es_prev = ema_slow.iloc[-2]
        hist = histogram.iloc[-1]
        hist_prev = histogram.iloc[-2]
        current_atr = atr.iloc[-1]

        if pd.isna(ef) or pd.isna(hist) or pd.isna(current_atr) or current_atr == 0:
            return None

        # BUY: EMA fast crosses above EMA slow
        if ef_prev <= es_prev and ef > es:
            if hist > 0 and hist_prev <= 0:
                strength = 0.9
            elif hist > 0:
                strength = 0.7
            else:
                strength = 0.5

            sl = price - (current_atr * 2.0)
            tp = price + (current_atr * 3.0)
            logger.info(
                f"[Momentum] BUY signal on {epic} | "
                f"strength={strength} EMA_cross+MACD"
            )
            return Signal(
                epic=epic, direction="BUY",
                entry_price=price, stop_loss=sl, take_profit=tp,
                strategy_name=self.name, strength=strength,
            )

        # SELL: EMA fast crosses below EMA slow
        if ef_prev >= es_prev and ef < es:
            if hist < 0 and hist_prev >= 0:
                strength = 0.9
            elif hist < 0:
                strength = 0.7
            else:
                strength = 0.5

            sl = price + (current_atr * 2.0)
            tp = price - (current_atr * 3.0)
            logger.info(
                f"[Momentum] SELL signal on {epic} | "
                f"strength={strength} EMA_cross+MACD"
            )
            return Signal(
                epic=epic, direction="SELL",
                entry_price=price, stop_loss=sl, take_profit=tp,
                strategy_name=self.name, strength=strength,
            )

        return None
