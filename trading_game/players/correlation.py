"""Statistical player: weights from the information coefficient (IC) of each
indicator signal against forward returns on the training set; the SIGN of
the IC becomes the indicator's orientation (learned interpretation)."""
import numpy as np
import pandas as pd

from trading_game.players.base import BasePlayer


class CorrelationPlayer(BasePlayer):
    name = "correlation_ic"
    method_description = (
        "Information-coefficient statistics: weight_i ~ |corr(signal_i, "
        "forward 21d return)| pooled across the universe; orientation_i = "
        "sign of that correlation."
    )

    HORIZON = 21

    def __init__(self, player_id, config, seed=0):
        super().__init__(player_id, config, seed)
        self.indicators = [
            "SMA_RATIO", "EMA_CROSS", "MACD_HIST", "ADX_TREND", "AROON",
            "RSI", "STOCH", "ROC", "CCI", "MFI",
            "BB_POSITION", "OBV_TREND", "VWAP_VALUE", "CMF",
        ]

    def fit(self, train_data):
        ics = {}
        for indicator in self.indicators:
            signal_all = []
            fwd_all = []
            for symbol, bundle in train_data.items():
                sig = bundle["signals"].get(indicator)
                if sig is None:
                    continue
                fwd = self.forward_returns(bundle["close"], self.HORIZON)
                joined = pd.concat([sig, fwd], axis=1).dropna()
                if len(joined) < 100:
                    continue
                signal_all.append(joined.iloc[:, 0].to_numpy())
                fwd_all.append(joined.iloc[:, 1].to_numpy())
            if not signal_all:
                ics[indicator] = 0.0
                continue
            s = np.concatenate(signal_all)
            f = np.concatenate(fwd_all)
            if np.std(s) == 0 or np.std(f) == 0:
                ics[indicator] = 0.0
                continue
            ics[indicator] = float(np.corrcoef(s, f)[0, 1])

        self.orientation = {k: (1.0 if v >= 0 else -1.0) for k, v in ics.items()}
        self.weights = self._normalize({k: abs(v) for k, v in ics.items()})
