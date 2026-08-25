import pandas as pd


class Indicators:

    @staticmethod
    def rsi(close: pd.Series, period: int = 14) -> pd.Series:
        delta = close.diff()
        gain = delta.where(delta > 0, 0.0).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0.0)).rolling(period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    @staticmethod
    def ema(data: pd.Series, period: int) -> pd.Series:
        return data.ewm(span=period, adjust=False).mean()

    @staticmethod
    def sma(data: pd.Series, period: int) -> pd.Series:
        return data.rolling(period).mean()

    @staticmethod
    def macd(
        close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9,
    ) -> tuple:
        ema_fast = close.ewm(span=fast, adjust=False).mean()
        ema_slow = close.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram

    @staticmethod
    def bollinger_bands(
        close: pd.Series, period: int = 20, std_dev: float = 2.0,
    ) -> tuple:
        sma = close.rolling(period).mean()
        std = close.rolling(period).std()
        upper = sma + (std * std_dev)
        lower = sma - (std * std_dev)
        return upper, sma, lower

    @staticmethod
    def atr(
        high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14,
    ) -> pd.Series:
        tr1 = high - low
        tr2 = (high - close.shift()).abs()
        tr3 = (low - close.shift()).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(period).mean()
