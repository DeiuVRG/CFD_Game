import logging
from typing import Optional

import pandas as pd

from config.settings import STRATEGY
from data.indicators import Indicators
from engine.signal import Signal
from strategies.base_strategy import BaseStrategy

logger = logging.getLogger(__name__)


class MomentumStrategy(BaseStrategy):
    """MACD + EMA crossover momentum strategy on 15-minute candles."""

    def __init__(self):
        super().__init__(name="Momentum", timeframe="MINUTE_15")

    def get_required_candles(self) -> int:
        return STRATEGY.MOM_MACD_SLOW + STRATEGY.MOM_MACD_SIGNAL + 10

    def analyze(self, epic: str, df: pd.DataFrame) -> Optional[Signal]:
        if len(df) < self.get_required_candles():
            return None

        close = df["close"]
        high = df["high"]
        low = df["low"]

        ema_fast = Indicators.ema(close, STRATEGY.MOM_EMA_FAST)
        ema_slow = Indicators.ema(close, STRATEGY.MOM_EMA_SLOW)
        _, _, histogram = Indicators.macd(
            close, STRATEGY.MOM_MACD_FAST, STRATEGY.MOM_MACD_SLOW,
            STRATEGY.MOM_MACD_SIGNAL,
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

        # BUY: EMA fast crosses above EMA slow (classic crossover)
        if ef_prev <= es_prev and ef > es:
            if hist > 0 and hist_prev <= 0:
                strength = 0.9  # Strong: MACD also crossing bullish
            elif hist > 0:
                strength = 0.7  # Moderate: MACD already positive
            else:
                strength = 0.5  # Weak: MACD still negative

            sl = price - (current_atr * STRATEGY.MOM_ATR_SL)
            tp = price + (current_atr * STRATEGY.MOM_ATR_TP)
            return Signal(
                epic=epic, direction="BUY",
                entry_price=price, stop_loss=sl, take_profit=tp,
                strategy_name=self.name, strength=strength,
            )

        # BUY (alt): EMA fast already above slow AND MACD histogram crosses positive
        if ef > es and hist > 0 and hist_prev <= 0:
            sl = price - (current_atr * STRATEGY.MOM_ATR_SL)
            tp = price + (current_atr * STRATEGY.MOM_ATR_TP)
            return Signal(
                epic=epic, direction="BUY",
                entry_price=price, stop_loss=sl, take_profit=tp,
                strategy_name=self.name, strength=0.65,
            )

        # SELL: EMA fast crosses below EMA slow
        if ef_prev >= es_prev and ef < es:
            if hist < 0 and hist_prev >= 0:
                strength = 0.9
            elif hist < 0:
                strength = 0.7
            else:
                strength = 0.5

            sl = price + (current_atr * STRATEGY.MOM_ATR_SL)
            tp = price - (current_atr * STRATEGY.MOM_ATR_TP)
            return Signal(
                epic=epic, direction="SELL",
                entry_price=price, stop_loss=sl, take_profit=tp,
                strategy_name=self.name, strength=strength,
            )

        # SELL (alt): EMA fast already below slow AND MACD histogram crosses negative
        if ef < es and hist < 0 and hist_prev >= 0:
            sl = price + (current_atr * STRATEGY.MOM_ATR_SL)
            tp = price - (current_atr * STRATEGY.MOM_ATR_TP)
            return Signal(
                epic=epic, direction="SELL",
                entry_price=price, stop_loss=sl, take_profit=tp,
                strategy_name=self.name, strength=0.65,
            )

        return None
