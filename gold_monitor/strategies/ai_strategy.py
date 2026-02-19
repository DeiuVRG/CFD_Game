import logging
from typing import Optional

import pandas as pd

from ai.feature_engineer import FeatureEngineer
from ai.model import GoldPredictor
from config.settings import AI, STRATEGY
from data.indicators import Indicators
from engine.signal import Signal
from strategies.base_strategy import BaseStrategy

logger = logging.getLogger(__name__)


class AIStrategy(BaseStrategy):
    """XGBoost AI strategy - predicts BUY/SELL/HOLD with confidence."""

    def __init__(self, model_path: str = None):
        super().__init__(name="AI", timeframe="MINUTE_5")
        self.predictor = GoldPredictor(model_path=model_path)
        self._loaded = self.predictor.load()
        if not self._loaded:
            logger.warning(f"AI model not loaded from {model_path or 'default'}. Run --train first.")

    def get_required_candles(self) -> int:
        return 60  # Need enough for indicators + features

    def analyze(self, epic: str, df: pd.DataFrame) -> Optional[Signal]:
        if not self.predictor.is_ready:
            return None

        if len(df) < self.get_required_candles():
            return None

        # Create features
        features = FeatureEngineer.create_features(df)
        if features.empty or features.iloc[-1].isna().any():
            return None

        # Predict
        signal_val, confidence, probs = self.predictor.predict(features)

        if confidence < AI.CONFIDENCE_THRESHOLD:
            return None

        if signal_val == 0:  # HOLD
            return None

        price = df["close"].iloc[-1]
        atr = Indicators.atr(df["high"], df["low"], df["close"], 14).iloc[-1]

        if pd.isna(atr) or atr == 0:
            return None

        if signal_val == 1:  # BUY
            direction = "BUY"
            sl = price - (atr * STRATEGY.SCALP_ATR_SL)
            tp = price + (atr * STRATEGY.SCALP_ATR_TP)
        else:  # SELL
            direction = "SELL"
            sl = price + (atr * STRATEGY.SCALP_ATR_SL)
            tp = price - (atr * STRATEGY.SCALP_ATR_TP)

        logger.info(
            f"[AI] {direction} signal on {epic} | "
            f"conf={confidence:.1%} probs={probs}"
        )

        return Signal(
            epic=epic,
            direction=direction,
            entry_price=price,
            stop_loss=sl,
            take_profit=tp,
            strategy_name=f"AI ({confidence:.0%})",
            strength=confidence,
        )
