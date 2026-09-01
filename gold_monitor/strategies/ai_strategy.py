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
    """XGBoost AI strategy - predicts BUY/SELL/HOLD with confidence.

    v3 execution model: analyze() must be fed TRAIN_INTERVAL (1h) candles -
    the same timeframe the model was trained and backtested on. SL/TP and the
    confidence threshold come from the per-instrument config when set.
    """

    def __init__(self, model_path: str = None, instrument=None):
        super().__init__(name="AI", timeframe="HOUR")
        self.instrument = instrument
        self.predictor = GoldPredictor(model_path=model_path)
        self._loaded = self.predictor.load()
        # Last prediction (for Discord / signals.db), set by analyze()
        self.last_confidence: float = 0.0
        self.last_probs: dict = {}
        if not self._loaded:
            logger.warning(f"AI model not loaded from {model_path or 'default'}. Run --train first.")

    def _sl_atr(self) -> float:
        return self.instrument.sl_atr() if self.instrument else STRATEGY.SCALP_ATR_SL

    def _tp_atr(self) -> float:
        return self.instrument.tp_atr() if self.instrument else STRATEGY.SCALP_ATR_TP

    def _confidence_threshold(self) -> float:
        if self.instrument:
            return self.instrument.confidence_threshold()
        return AI.CONFIDENCE_THRESHOLD

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
        self.last_confidence = float(confidence)
        self.last_probs = dict(probs) if probs else {}

        if confidence < self._confidence_threshold():
            return None

        if signal_val == 0:  # HOLD
            return None

        price = df["close"].iloc[-1]
        atr = Indicators.atr(df["high"], df["low"], df["close"], 14).iloc[-1]

        if pd.isna(atr) or atr == 0:
            return None

        if signal_val == 1:  # BUY
            direction = "BUY"
            sl = price - (atr * self._sl_atr())
            tp = price + (atr * self._tp_atr())
        else:  # SELL
            direction = "SELL"
            sl = price + (atr * self._sl_atr())
            tp = price - (atr * self._tp_atr())

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
