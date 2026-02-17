import numpy as np
import pandas as pd

from data.indicators import Indicators


class FeatureEngineer:
    """Creates 20 features from OHLCV price data for the XGBoost model."""

    @staticmethod
    def create_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Input: DataFrame with columns [timestamp, open, high, low, close, volume]
        Output: DataFrame with 20 engineered features (NaN rows dropped)
        """
        close = df["close"]
        high = df["high"]
        low = df["low"]
        open_p = df["open"]
        volume = df["volume"] if "volume" in df.columns else pd.Series(0, index=df.index)

        features = pd.DataFrame(index=df.index)

        # 1-2: RSI (momentum)
        features["rsi_14"] = Indicators.rsi(close, 14)
        features["rsi_7"] = Indicators.rsi(close, 7)

        # 3-4: MACD (trend)
        macd_line, signal_line, histogram = Indicators.macd(close, 12, 26, 9)
        features["macd_hist"] = histogram
        features["macd_signal_dist"] = macd_line - signal_line

        # 5-6: Bollinger Bands (volatility + position)
        bb_upper, bb_middle, bb_lower = Indicators.bollinger_bands(close, 20, 2.0)
        bb_range = bb_upper - bb_lower
        features["bb_position"] = np.where(
            bb_range != 0,
            (close - bb_lower) / bb_range,
            0.5,
        )
        features["bb_width"] = np.where(
            bb_middle != 0,
            bb_range / bb_middle,
            0,
        )

        # 7: ATR (volatility)
        features["atr_14"] = Indicators.atr(high, low, close, 14)

        # 8: EMA ratio (trend)
        ema9 = Indicators.ema(close, 9)
        ema21 = Indicators.ema(close, 21)
        features["ema_ratio"] = np.where(ema21 != 0, ema9 / ema21, 1.0)

        # 9-11: Price changes (momentum)
        features["price_change_5"] = close.pct_change(5)
        features["price_change_10"] = close.pct_change(10)
        features["price_change_30"] = close.pct_change(30)

        # 12: High-Low range normalized
        features["high_low_range"] = np.where(
            close != 0,
            (high - low) / close,
            0,
        )

        # 13: Close vs Open (candle direction)
        features["close_vs_open"] = np.where(
            open_p != 0,
            (close - open_p) / open_p,
            0,
        )

        # 14: Volume change vs 20-period average
        vol_sma = volume.rolling(20).mean()
        features["volume_change"] = np.where(
            vol_sma != 0,
            volume / vol_sma,
            1.0,
        )

        # 15-17: Temporal features
        if "timestamp" in df.columns:
            ts = pd.to_datetime(df["timestamp"])
            hour = ts.dt.hour + ts.dt.minute / 60
            features["hour_sin"] = np.sin(2 * np.pi * hour / 24)
            features["hour_cos"] = np.cos(2 * np.pi * hour / 24)
            features["day_of_week"] = ts.dt.dayofweek
        else:
            features["hour_sin"] = 0
            features["hour_cos"] = 0
            features["day_of_week"] = 0

        # 18-19: Candle shadows (rejection signals)
        candle_range = high - low
        body_top = pd.concat([close, open_p], axis=1).max(axis=1)
        body_bottom = pd.concat([close, open_p], axis=1).min(axis=1)
        features["upper_shadow"] = np.where(
            candle_range != 0,
            (high - body_top) / candle_range,
            0,
        )
        features["lower_shadow"] = np.where(
            candle_range != 0,
            (body_bottom - low) / candle_range,
            0,
        )

        # 20: Consecutive direction
        direction = (close.diff() > 0).astype(int)
        groups = (direction != direction.shift()).cumsum()
        features["consecutive_direction"] = direction.groupby(groups).cumcount() + 1
        # Make it negative for consecutive down candles
        features["consecutive_direction"] = features["consecutive_direction"] * np.where(
            direction == 1, 1, -1
        )

        return features

    @staticmethod
    def create_labels(df: pd.DataFrame, horizon: int = 6, threshold: float = 0.003) -> pd.Series:
        """
        Create labels: BUY (1), SELL (-1), HOLD (0)
        based on future price change over 'horizon' candles.
        threshold: minimum % change for a signal (0.3%)
        """
        close = df["close"]
        future_close = close.shift(-horizon)
        pct_change = (future_close - close) / close

        labels = pd.Series(0, index=df.index, dtype=int)
        labels[pct_change > threshold] = 1    # BUY
        labels[pct_change < -threshold] = -1  # SELL
        # Last 'horizon' rows will have NaN future - mark as 0
        labels.iloc[-horizon:] = 0

        return labels
