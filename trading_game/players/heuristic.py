"""Heuristic players: equal-weight baseline, momentum, mean-reversion."""
from trading_game.players.base import BasePlayer


class EqualWeightPlayer(BasePlayer):
    name = "equal_weight"
    method_description = (
        "Baseline: equal weights on 5 classic indicators, canonical "
        "interpretations. Every other method must beat this to matter."
    )

    def __init__(self, player_id, config, seed=0):
        super().__init__(player_id, config, seed)
        self.indicators = ["RSI", "MACD_HIST", "BB_POSITION",
                           "ADX_TREND", "CMF"]

    def fit(self, train_data):
        n = len(self.indicators)
        self.weights = {k: 1.0 / n for k in self.indicators}


class MomentumPlayer(BasePlayer):
    name = "momentum"
    method_description = (
        "Experience-based heuristic: heavy trend/momentum weights; "
        "oscillators flipped from contrarian to trend-following "
        "(overbought = strength, not a fade)."
    )

    def __init__(self, player_id, config, seed=0):
        super().__init__(player_id, config, seed)
        self.indicators = ["SMA_RATIO", "EMA_CROSS", "MACD_HIST",
                           "ADX_TREND", "AROON", "ROC", "RSI",
                           "OBV_TREND", "VOLUME_SURGE"]
        # Flip the contrarian oscillator so high RSI reads as strength
        self.orientation = {"RSI": -1.0}

    def fit(self, train_data):
        raw = {
            "SMA_RATIO": 0.16, "EMA_CROSS": 0.16, "MACD_HIST": 0.14,
            "ADX_TREND": 0.12, "AROON": 0.10, "ROC": 0.12, "RSI": 0.06,
            "OBV_TREND": 0.08, "VOLUME_SURGE": 0.06,
        }
        self.weights = self._normalize(raw)


class MeanReversionPlayer(BasePlayer):
    name = "mean_reversion"
    method_description = (
        "Experience-based heuristic: contrarian oscillator stack "
        "(buy oversold, sell overbought) with band-position confirmation."
    )

    def __init__(self, player_id, config, seed=0):
        super().__init__(player_id, config, seed)
        self.indicators = ["RSI", "STOCH", "WILLR", "CCI", "MFI",
                           "BB_POSITION", "KELTNER_POSITION", "VWAP_VALUE"]

    def fit(self, train_data):
        raw = {
            "RSI": 0.18, "STOCH": 0.13, "WILLR": 0.11, "CCI": 0.12,
            "MFI": 0.11, "BB_POSITION": 0.16, "KELTNER_POSITION": 0.10,
            "VWAP_VALUE": 0.09,
        }
        self.weights = self._normalize(raw)
