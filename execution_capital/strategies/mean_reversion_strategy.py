import logging
from typing import Optional

import pandas as pd

from config.settings import MEAN_REVERSION, RISK
from data.indicators import Indicators
from engine.signal import Signal
from strategies.base_strategy import BaseStrategy

logger = logging.getLogger(__name__)


class MeanReversionStrategy(BaseStrategy):
    """Bollinger Bands + RSI mean reversion in ranging markets.

    Only trades when BB width is below average (market is ranging).
    BUY:  price below lower BB + RSI < 25 (deeply oversold)
    SELL: price above upper BB + RSI > 75 (deeply overbought)
    Target: middle Bollinger Band
    """

    def __init__(self):
        super().__init__(name="MeanReversion", timeframe=MEAN_REVERSION.TIMEFRAME)

    def get_required_candles(self) -> int:
        return max(
            MEAN_REVERSION.BB_PERIOD,
            MEAN_REVERSION.RSI_PERIOD,
            MEAN_REVERSION.BB_WIDTH_LOOKBACK,
        ) + 10

    def analyze(self, epic: str, df: pd.DataFrame) -> Optional[Signal]:
        if len(df) < self.get_required_candles():
            return None

        close = df["close"]
        high = df["high"]
        low = df["low"]

        bb_upper, bb_middle, bb_lower = Indicators.bollinger_bands(
            close, MEAN_REVERSION.BB_PERIOD, MEAN_REVERSION.BB_STD,
        )
        rsi = Indicators.rsi(close, MEAN_REVERSION.RSI_PERIOD)
        atr = Indicators.atr(high, low, close)

        price = close.iloc[-1]
        upper = bb_upper.iloc[-1]
        middle = bb_middle.iloc[-1]
        lower = bb_lower.iloc[-1]
        rsi_now = rsi.iloc[-1]
        current_atr = atr.iloc[-1]

        if pd.isna(rsi_now) or pd.isna(middle) or pd.isna(current_atr) or current_atr == 0:
            return None

        # Check if market is ranging (not trending)
        bb_width = (bb_upper - bb_lower) / bb_middle
        avg_width = bb_width.rolling(MEAN_REVERSION.BB_WIDTH_LOOKBACK).mean().iloc[-1]
        current_width = bb_width.iloc[-1]

        if pd.isna(avg_width):
            return None

        if current_width > avg_width * MEAN_REVERSION.TRENDING_MULTIPLIER:
            return None  # Market is trending, skip

        # BUY: price below lower BB + RSI deeply oversold
        if price < lower and rsi_now < MEAN_REVERSION.RSI_OVERSOLD:
            sl = price - (current_atr * 1.0)
            tp = middle
            rr = abs(tp - price) / abs(price - sl) if abs(price - sl) > 0 else 0
            if rr >= RISK.MIN_RISK_REWARD:
                logger.info(
                    f"[MeanRev] BUY signal on {epic} | "
                    f"RSI={rsi_now:.1f} price_below_lower_BB"
                )
                return Signal(
                    epic=epic, direction="BUY",
                    entry_price=price, stop_loss=sl, take_profit=tp,
                    strategy_name=self.name, strength=0.8,
                )

        # SELL: price above upper BB + RSI deeply overbought
        if price > upper and rsi_now > MEAN_REVERSION.RSI_OVERBOUGHT:
            sl = price + (current_atr * 1.0)
            tp = middle
            rr = abs(price - tp) / abs(sl - price) if abs(sl - price) > 0 else 0
            if rr >= RISK.MIN_RISK_REWARD:
                logger.info(
                    f"[MeanRev] SELL signal on {epic} | "
                    f"RSI={rsi_now:.1f} price_above_upper_BB"
                )
                return Signal(
                    epic=epic, direction="SELL",
                    entry_price=price, stop_loss=sl, take_profit=tp,
                    strategy_name=self.name, strength=0.8,
                )

        return None
