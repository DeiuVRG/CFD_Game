import logging
from typing import Optional

import pandas as pd

from config.settings import SCALPING, RISK
from data.indicators import Indicators
from engine.signal import Signal
from strategies.base_strategy import BaseStrategy

logger = logging.getLogger(__name__)


class ScalpingStrategy(BaseStrategy):
    """RSI + Bollinger Bands scalping on 5-minute candles.

    BUY:  RSI crosses above oversold (30) + price near lower BB
    SELL: RSI crosses below overbought (70) + price near upper BB
    Target: middle Bollinger Band (mean reversion to center)
    """

    def __init__(self):
        super().__init__(name="Scalping", timeframe=SCALPING.TIMEFRAME)

    def get_required_candles(self) -> int:
        return max(SCALPING.RSI_PERIOD, SCALPING.BB_PERIOD) + 30

    def analyze(self, epic: str, df: pd.DataFrame) -> Optional[Signal]:
        if len(df) < self.get_required_candles():
            return None

        close = df["close"]
        high = df["high"]
        low = df["low"]

        rsi = Indicators.rsi(close, SCALPING.RSI_PERIOD)
        bb_upper, bb_middle, bb_lower = Indicators.bollinger_bands(
            close, SCALPING.BB_PERIOD, SCALPING.BB_STD,
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

        # BUY: RSI crosses above oversold + price near lower BB
        if rsi_prev < SCALPING.RSI_OVERSOLD and rsi_now >= SCALPING.RSI_OVERSOLD:
            if price <= lower * 1.002:
                sl = price - (current_atr * RISK.SL_ATR_MULTIPLIER)
                tp = middle
                rr = abs(tp - price) / abs(price - sl) if abs(price - sl) > 0 else 0
                if rr >= RISK.MIN_RISK_REWARD:
                    logger.info(f"[Scalping] BUY signal on {epic} | RSI={rsi_now:.1f}")
                    return Signal(
                        epic=epic, direction="BUY",
                        entry_price=price, stop_loss=sl, take_profit=tp,
                        strategy_name=self.name, strength=0.7,
                    )

        # SELL: RSI crosses below overbought + price near upper BB
        if rsi_prev > SCALPING.RSI_OVERBOUGHT and rsi_now <= SCALPING.RSI_OVERBOUGHT:
            if price >= upper * 0.998:
                sl = price + (current_atr * RISK.SL_ATR_MULTIPLIER)
                tp = middle
                rr = abs(price - tp) / abs(sl - price) if abs(sl - price) > 0 else 0
                if rr >= RISK.MIN_RISK_REWARD:
                    logger.info(f"[Scalping] SELL signal on {epic} | RSI={rsi_now:.1f}")
                    return Signal(
                        epic=epic, direction="SELL",
                        entry_price=price, stop_loss=sl, take_profit=tp,
                        strategy_name=self.name, strength=0.7,
                    )

        return None
