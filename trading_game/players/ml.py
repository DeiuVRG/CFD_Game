"""Machine-learning players: weights from feature importances of models
trained to predict forward returns from indicator signals. CPU-friendly by
design; both adapt automatically to whatever hardware runs them (the spec's
GPU acceleration is optional - these models don't need it)."""
import numpy as np
import pandas as pd

from trading_game.players.base import BasePlayer


def build_pooled_dataset(train_data, indicators, horizon):
    """Pooled (X=signals, y=forward return) across the whole universe."""
    xs, ys = [], []
    for symbol, bundle in train_data.items():
        sig = bundle["signals"]
        cols = [c for c in indicators if c in sig.columns]
        fwd = BasePlayer.forward_returns(bundle["close"], horizon)
        joined = pd.concat([sig[cols], fwd.rename("__fwd")], axis=1).dropna()
        if len(joined) < 100:
            continue
        xs.append(joined[cols].to_numpy())
        ys.append(joined["__fwd"].to_numpy())
    if not xs:
        return None, None, []
    return np.vstack(xs), np.concatenate(ys), cols


def orientation_from_correlation(X, y, indicators):
    orient = {}
    for i, indicator in enumerate(indicators):
        col = X[:, i]
        if np.std(col) == 0:
            orient[indicator] = 1.0
            continue
        orient[indicator] = 1.0 if np.corrcoef(col, y)[0, 1] >= 0 else -1.0
    return orient


class RandomForestPlayer(BasePlayer):
    name = "random_forest"
    method_description = (
        "RandomForestRegressor on pooled (indicator signals -> forward 21d "
        "return); weights = normalized feature importances; orientation = "
        "sign of each signal's correlation with forward returns."
    )

    HORIZON = 21

    def __init__(self, player_id, config, seed=0):
        super().__init__(player_id, config, seed)
        self.indicators = [
            "SMA_RATIO", "EMA_CROSS", "MACD_HIST", "ADX_TREND",
            "RSI", "STOCH", "ROC", "MFI",
            "BB_POSITION", "ATR_REGIME", "OBV_TREND", "CMF",
        ]

    def fit(self, train_data):
        from sklearn.ensemble import RandomForestRegressor

        X, y, cols = build_pooled_dataset(train_data, self.indicators, self.HORIZON)
        if X is None:
            self.weights = self._normalize({k: 1.0 for k in self.indicators})
            return
        # Subsample for speed; RF importances stabilize quickly
        rng = np.random.default_rng(self.seed)
        if len(X) > 30000:
            idx = rng.choice(len(X), 30000, replace=False)
            X, y = X[idx], y[idx]
        model = RandomForestRegressor(
            n_estimators=120, max_depth=6, min_samples_leaf=50,
            random_state=self.seed, n_jobs=-1,
        )
        model.fit(X, y)
        self.orientation = orientation_from_correlation(X, y, cols)
        self.weights = self._normalize(dict(zip(cols, model.feature_importances_)))


class XGBoostPlayer(BasePlayer):
    name = "xgboost"
    method_description = (
        "XGBoost regressor on pooled (indicator signals -> forward 10d "
        "return); weights = normalized gain importances; orientation from "
        "signal/return correlation. Shorter horizon than the RF player."
    )

    HORIZON = 10

    def __init__(self, player_id, config, seed=0):
        super().__init__(player_id, config, seed)
        self.indicators = [
            "EMA_CROSS", "MACD_HIST", "AROON", "ADX_TREND",
            "RSI", "WILLR", "CCI", "ROC",
            "KELTNER_POSITION", "VOLUME_SURGE", "VWAP_VALUE",
        ]

    def fit(self, train_data):
        from xgboost import XGBRegressor

        X, y, cols = build_pooled_dataset(train_data, self.indicators, self.HORIZON)
        if X is None:
            self.weights = self._normalize({k: 1.0 for k in self.indicators})
            return
        model = XGBRegressor(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=1.0,
            random_state=self.seed, verbosity=0, n_jobs=-1,
        )
        model.fit(X, y)
        self.orientation = orientation_from_correlation(X, y, cols)
        importances = model.feature_importances_
        self.weights = self._normalize(dict(zip(cols, importances)))
