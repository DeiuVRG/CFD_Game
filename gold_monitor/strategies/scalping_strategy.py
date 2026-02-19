import logging
from typing import Optional

import pandas as pd

from config.settings import STRATEGY
from data.indicators import Indicators
from engine.signal import Signal
from strategies.base_strategy import BaseStrategy

logger = logging.getLogger(__name__)


class ScalpingStrategy(BaseStrategy):
    """RSI + Bollinger Bands scalping on 5-minute candles."""

    def __init__(self):
        super().__init__(name="Scalping", timeframe="MINUTE_5")

    def get_required_candles(self) -> int:
        return max(STRATEGY.SCALP_RSI_PERIOD, STRATEGY.SCALP_BB_PERIOD) + 30

    def analyze(self, epic: str, df: pd.DataFrame) -> Optional[Signal]:
        if len(df) < self.get_required_candles():
            return None

        close = df["close"]
        high = df["high"]
        low = df["low"]

        rsi = Indicators.rsi(close, STRATEGY.SCALP_RSI_PERIOD)
        bb_upper, bb_middle, bb_lower = Indicators.bollinger_bands(
            close, STRATEGY.SCALP_BB_PERIOD, STRATEGY.SCALP_BB_STD,
        )
        atr = Indicators.atr(high, low, close)

        price = close.iloc[-1]
        rsi_now = rsi.iloc[-1]
        rsi_prev = rsi.iloc[-2]
        lower = bb_lower.iloc[-1]
        upper = bb_upper.iloc[-1]
        middle = bb_middle.iloc[-1]
        current_atr = atr.iloc[-1]

        if pd.isna(rsi_now) or pd.isna(current_atr) or current_atr == 0:
            return None

        # BUY: RSI in oversold zone and turning up + price near lower BB
        if rsi_now < 35 and rsi_now > rsi_prev and price <= lower * 1.005:
            sl = price - (current_atr * STRATEGY.SCALP_ATR_SL)
            # TP = max(BB middle, 3x ATR) - optimized for larger targets
            tp_bb = middle
            tp_atr = price + (current_atr * STRATEGY.SCALP_ATR_TP)
            tp = max(tp_bb, tp_atr)

            rr = abs(tp - price) / abs(price - sl) if abs(price - sl) > 0 else 0
            if rr >= 1.5:  # Minimum 1.5:1 R:R (optimized)
                return Signal(
                    epic=epic, direction="BUY",
                    entry_price=price, stop_loss=sl, take_profit=tp,
                    strategy_name=self.name, strength=0.7,
                )

        # SELL: RSI in overbought zone and turning down + price near upper BB
        if rsi_now > 65 and rsi_now < rsi_prev and price >= upper * 0.995:
            sl = price + (current_atr * STRATEGY.SCALP_ATR_SL)
            tp_bb = middle
            tp_atr = price - (current_atr * STRATEGY.SCALP_ATR_TP)
            tp = min(tp_bb, tp_atr)

            rr = abs(price - tp) / abs(sl - price) if abs(sl - price) > 0 else 0
            if rr >= 1.5:
                return Signal(
                    epic=epic, direction="SELL",
                    entry_price=price, stop_loss=sl, take_profit=tp,
                    strategy_name=self.name, strength=0.7,
                )

        return None
