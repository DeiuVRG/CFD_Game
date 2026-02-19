import numpy as np
import pandas as pd

from data.indicators import Indicators


class FeatureEngineer:
    """Creates features from OHLCV price data for the XGBoost model.

    v2 improvements:
    - Removed dead features (volume_change, day_of_week)
    - Added volatility regime (ATR ratio, BB width change)
    - Added ADX for trend strength
    - Added Stochastic RSI
    - Normalized ATR relative to price
    - Better label creation with proper NaN handling
    """

    FEATURE_NAMES = [
        "rsi_14", "rsi_7", "stoch_rsi",
        "macd_hist", "macd_signal_dist", "macd_hist_change",
        "bb_position", "bb_width", "bb_width_change",
        "atr_pct", "atr_ratio",
        "ema_ratio", "ema50_slope", "adx",
        "price_change_5", "price_change_10",
        "high_low_range", "close_vs_open",
        "upper_shadow", "lower_shadow",
        "hour_sin", "hour_cos",
        "consecutive_direction",
    ]

    @staticmethod
    def create_features(df: pd.DataFrame) -> pd.DataFrame:
        close = df["close"]
        high = df["high"]
        low = df["low"]
        open_p = df["open"]

        features = pd.DataFrame(index=df.index)

        # --- MOMENTUM ---
        features["rsi_14"] = Indicators.rsi(close, 14)
        features["rsi_7"] = Indicators.rsi(close, 7)

        # Stochastic RSI
        rsi_14 = features["rsi_14"]
        rsi_min = rsi_14.rolling(14).min()
        rsi_max = rsi_14.rolling(14).max()
        rsi_range = rsi_max - rsi_min
        features["stoch_rsi"] = np.where(rsi_range != 0, (rsi_14 - rsi_min) / rsi_range, 0.5)

        # MACD
        macd_line, signal_line, histogram = Indicators.macd(close, 12, 26, 9)
        features["macd_hist"] = histogram
        features["macd_signal_dist"] = macd_line - signal_line
        features["macd_hist_change"] = histogram - histogram.shift(1)

        # --- VOLATILITY ---
        bb_upper, bb_middle, bb_lower = Indicators.bollinger_bands(close, 20, 2.0)
        bb_range = bb_upper - bb_lower
        features["bb_position"] = np.where(bb_range != 0, (close - bb_lower) / bb_range, 0.5)
        features["bb_width"] = np.where(bb_middle != 0, bb_range / bb_middle, 0)

        # BB width change (volatility expanding/contracting?)
        bb_w = features["bb_width"]
        bb_w_prev = bb_w.shift(5)
        features["bb_width_change"] = np.where(bb_w_prev != 0, (bb_w - bb_w_prev) / bb_w_prev, 0)

        # ATR normalized by price
        atr_14 = Indicators.atr(high, low, close, 14)
        features["atr_pct"] = np.where(close != 0, atr_14 / close, 0)

        # ATR ratio = current vs average (volatility regime)
        atr_avg = atr_14.rolling(20).mean()
        features["atr_ratio"] = np.where(atr_avg != 0, atr_14 / atr_avg, 1.0)

        # --- TREND ---
        ema9 = Indicators.ema(close, 9)
        ema21 = Indicators.ema(close, 21)
        features["ema_ratio"] = np.where(ema21 != 0, ema9 / ema21, 1.0)

        ema50 = Indicators.ema(close, 50)
        ema50_shifted = ema50.shift(5)
        features["ema50_slope"] = np.where(ema50_shifted != 0, (ema50 - ema50_shifted) / ema50_shifted, 0)

        features["adx"] = Indicators.adx(high, low, close, 14)

        # --- PRICE ACTION ---
        features["price_change_5"] = close.pct_change(5)
        features["price_change_10"] = close.pct_change(10)

        features["high_low_range"] = np.where(close != 0, (high - low) / close, 0)
        features["close_vs_open"] = np.where(open_p != 0, (close - open_p) / open_p, 0)

        # Candle shadows (rejection wicks)
        candle_range = high - low
        body_top = pd.concat([close, open_p], axis=1).max(axis=1)
        body_bottom = pd.concat([close, open_p], axis=1).min(axis=1)
        features["upper_shadow"] = np.where(candle_range != 0, (high - body_top) / candle_range, 0)
        features["lower_shadow"] = np.where(candle_range != 0, (body_bottom - low) / candle_range, 0)

        # Session (hour cyclical encoding)
        if "timestamp" in df.columns:
            ts = pd.to_datetime(df["timestamp"])
            hour = ts.dt.hour + ts.dt.minute / 60
            features["hour_sin"] = np.sin(2 * np.pi * hour / 24)
            features["hour_cos"] = np.cos(2 * np.pi * hour / 24)
        else:
            features["hour_sin"] = 0
            features["hour_cos"] = 0

        # Consecutive direction
        direction = (close.diff() > 0).astype(int)
        groups = (direction != direction.shift()).cumsum()
        consec = direction.groupby(groups).cumcount() + 1
        features["consecutive_direction"] = consec * np.where(direction == 1, 1, -1)

        return features

    @staticmethod
    def create_labels(
        df: pd.DataFrame,
        horizon: int = 6,
        threshold: float = 0.003,
    ) -> pd.Series:
        """
        Create labels: BUY (1), SELL (-1), HOLD (0).
        Uses future price change (shift(-horizon)) - this is correct for
        supervised learning labels. Anti-bias comes from walk-forward validation.
        """
        close = df["close"]
        future_close = close.shift(-horizon)
        pct_change = (future_close - close) / close

        labels = pd.Series(0, index=df.index, dtype=float)
        labels[pct_change > threshold] = 1
        labels[pct_change < -threshold] = -1
        labels.iloc[-horizon:] = np.nan  # Mark as NaN (will be dropped)

        return labels
