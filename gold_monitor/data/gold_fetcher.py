import logging
import time
from datetime import datetime, timezone
from typing import Optional

import pandas as pd
import requests
import yfinance as yf

from config.settings import InstrumentConfig, TWELVEDATA_API_KEY

# Suppress yfinance "possibly delisted" spam when market is closed
logging.getLogger("yfinance").setLevel(logging.CRITICAL)

logger = logging.getLogger(__name__)

TWELVEDATA_PRICE_URL = "https://api.twelvedata.com/price"
YAHOO_API_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
YAHOO_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
}

# Daily credit limit for Twelve Data free tier
TWELVEDATA_DAILY_LIMIT = 800


class CreditTracker:
    """Tracks Twelve Data API credits used today."""

    def __init__(self, daily_limit: int = TWELVEDATA_DAILY_LIMIT):
        self.daily_limit = daily_limit
        self._credits_used: int = 0
        self._reset_date: str = ""
        self._exhausted = False

    @property
    def credits_remaining(self) -> int:
        self._check_reset()
        return max(0, self.daily_limit - self._credits_used)

    @property
    def is_exhausted(self) -> bool:
        self._check_reset()
        return self._exhausted

    @property
    def source_name(self) -> str:
        return "Yahoo" if self.is_exhausted else "TwelveData"

    def use_credits(self, count: int):
        self._check_reset()
        self._credits_used += count
        if self._credits_used >= self.daily_limit:
            self._exhausted = True
            logger.warning(
                f"Twelve Data credits exhausted ({self._credits_used}/{self.daily_limit}). "
                f"Switching to Yahoo Finance."
            )

    def mark_exhausted(self):
        """Called when API returns rate limit error."""
        self._exhausted = True
        self._credits_used = self.daily_limit

    def _check_reset(self):
        """Reset credits at midnight UTC."""
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if today != self._reset_date:
            self._reset_date = today
            self._credits_used = 0
            self._exhausted = False
            logger.info("Twelve Data credits reset for new day.")


# Global credit tracker (shared across all fetchers)
credit_tracker = CreditTracker()


def fetch_all_prices_batch(instruments: list) -> dict:
    """
    Fetch prices for ALL instruments in a single API call.
    Uses Twelve Data if credits available, otherwise Yahoo Finance.
    Returns dict: { 'XAU/USD': 5025.45, 'EUR/USD': 1.1866, ... }
    """
    enabled = [i for i in instruments if i.TWELVEDATA_SYMBOL and i.ENABLED]
    if not enabled:
        return {}

    # Try Twelve Data first (if credits available)
    if not credit_tracker.is_exhausted and TWELVEDATA_API_KEY:
        prices = _fetch_twelvedata_batch(enabled)
        if prices:
            return prices

    # Fallback: Yahoo Finance
    return _fetch_yahoo_batch(enabled)


def _fetch_twelvedata_batch(instruments: list) -> dict:
    """Batch fetch from Twelve Data API."""
    td_symbols = [i.TWELVEDATA_SYMBOL for i in instruments]

    try:
        resp = requests.get(
            TWELVEDATA_PRICE_URL,
            params={"symbol": ",".join(td_symbols), "apikey": TWELVEDATA_API_KEY},
            timeout=5,
        )

        if resp.status_code != 200:
            return {}

        data = resp.json()

        if "code" in data:
            code = data.get("code")
            if code == 429:
                # Rate limit hit
                credit_tracker.mark_exhausted()
                logger.warning("Twelve Data rate limit hit, switching to Yahoo Finance")
            return {}

        # Track credits used (1 per symbol)
        credit_tracker.use_credits(len(td_symbols))

        prices = {}
        if len(td_symbols) == 1:
            p = data.get("price")
            if p:
                prices[td_symbols[0]] = float(p)
        else:
            for sym in td_symbols:
                entry = data.get(sym, {})
                p = entry.get("price")
                if p:
                    prices[sym] = float(p)

        return prices

    except Exception as e:
        logger.error(f"Twelve Data batch error: {e}")
        return {}


def _fetch_yahoo_batch(instruments: list) -> dict:
    """Fetch prices from Yahoo Finance direct API (fallback)."""
    prices = {}
    for inst in instruments:
        try:
            url = YAHOO_API_URL.format(symbol=inst.SYMBOL)
            resp = requests.get(
                url,
                headers=YAHOO_HEADERS,
                params={"interval": "1m", "range": "1d"},
                timeout=5,
            )

            if resp.status_code != 200:
                continue

            data = resp.json()
            result = data.get("chart", {}).get("result", [])
            if not result:
                continue

            meta = result[0].get("meta", {})
            price = meta.get("regularMarketPrice", 0)
            if price:
                prices[inst.TWELVEDATA_SYMBOL] = float(price)

        except Exception:
            continue

    return prices


class MarketFetcher:
    """Fetches live prices via Twelve Data / Yahoo Finance, candles via yfinance."""

    def __init__(self, instrument: InstrumentConfig):
        self.instrument = instrument
        # Cache
        self._last_price: float = 0
        self._last_prev_close: float = 0
        self._last_change: float = 0
        self._last_change_pct: float = 0

    def update_from_batch(self, price: float):
        """Update price from batch fetch result."""
        if price <= 0:
            return

        prev_close = self._last_prev_close if self._last_prev_close > 0 else price
        change = price - prev_close
        change_pct = (change / prev_close * 100) if prev_close else 0

        self._last_price = round(price, 5)
        self._last_change = round(change, 5)
        self._last_change_pct = round(change_pct, 3)

    def get_live_price(self) -> dict:
        """Return cached price (updated via batch fetch from engine)."""
        if self._last_price > 0:
            return {
                "price": self._last_price,
                "prev_close": self._last_prev_close,
                "change": self._last_change,
                "change_pct": self._last_change_pct,
            }
        # First call - fetch individually to get prev_close
        return self._fetch_initial()

    def _fetch_initial(self) -> dict:
        """Fetch price + prev_close at startup."""
        td_symbol = self.instrument.TWELVEDATA_SYMBOL

        # Try Twelve Data /quote for prev_close
        if td_symbol and TWELVEDATA_API_KEY and not credit_tracker.is_exhausted:
            try:
                resp = requests.get(
                    "https://api.twelvedata.com/quote",
                    params={"symbol": td_symbol, "apikey": TWELVEDATA_API_KEY},
                    timeout=5,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    if "code" not in data:
                        credit_tracker.use_credits(1)
                        price = float(data.get("close", 0))
                        prev_close = float(data.get("previous_close", 0))
                        if price:
                            return self._update_and_return(price, prev_close)
            except Exception:
                pass

        # Fallback: Yahoo Finance
        try:
            ticker = yf.Ticker(self.instrument.SYMBOL)
            df = ticker.history(period="5d", interval="1d")
            if not df.empty:
                price = float(df["Close"].iloc[-1])
                prev_close = float(df["Close"].iloc[-2]) if len(df) > 1 else float(df["Open"].iloc[0])
                return self._update_and_return(price, prev_close)
        except Exception:
            pass

        return {"price": 0, "prev_close": 0, "change": 0, "change_pct": 0}

    def _update_and_return(self, price: float, prev_close: float) -> dict:
        change = price - prev_close if prev_close else 0
        change_pct = (change / prev_close * 100) if prev_close else 0

        self._last_price = round(price, 5)
        self._last_prev_close = round(prev_close, 5)
        self._last_change = round(change, 5)
        self._last_change_pct = round(change_pct, 3)

        return {
            "price": self._last_price,
            "prev_close": self._last_prev_close,
            "change": self._last_change,
            "change_pct": self._last_change_pct,
        }

    def get_candles(
        self,
        period: str = None,
        interval: str = None,
        count: int = None,
    ) -> pd.DataFrame:
        """Get OHLCV candles via yfinance (for AI analysis)."""
        period = period or self.instrument.HISTORY_PERIOD
        interval = interval or self.instrument.CANDLE_INTERVAL

        try:
            ticker = yf.Ticker(self.instrument.SYMBOL)
            df = ticker.history(period=period, interval=interval)
            if df.empty:
                return pd.DataFrame()

            df = df.reset_index()
            df.columns = [c.lower().replace(" ", "_") for c in df.columns]

            if "datetime" in df.columns:
                df.rename(columns={"datetime": "timestamp"}, inplace=True)
            elif "date" in df.columns:
                df.rename(columns={"date": "timestamp"}, inplace=True)

            required = ["timestamp", "open", "high", "low", "close"]
            for col in required:
                if col not in df.columns:
                    return pd.DataFrame()

            df.sort_values("timestamp", inplace=True)
            df.reset_index(drop=True, inplace=True)

            if count and len(df) > count:
                df = df.tail(count).reset_index(drop=True)

            return df

        except Exception as e:
            logger.error(f"Failed to get candles for {self.instrument.SYMBOL}: {e}")
            return pd.DataFrame()

    def get_training_data(self) -> pd.DataFrame:
        """Get historical data for AI training."""
        return self.get_candles(
            period=self.instrument.TRAIN_PERIOD,
            interval=self.instrument.TRAIN_INTERVAL,
        )


# Backwards compatibility alias
GoldFetcher = MarketFetcher
