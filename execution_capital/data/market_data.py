import logging
import pandas as pd

from broker.capital_client import CapitalClient
from broker.session_manager import SessionManager

logger = logging.getLogger(__name__)


class MarketDataFetcher:
    def __init__(self, session_manager: SessionManager, client: CapitalClient):
        self.session_manager = session_manager
        self.client = client

    def get_candles(
        self, epic: str, resolution: str, count: int = 100,
    ) -> pd.DataFrame:
        self.session_manager.ensure_session()
        bars = self.client.get_prices(epic, resolution, max_bars=count)

        if not bars:
            logger.warning(f"No candle data for {epic} ({resolution})")
            return pd.DataFrame()

        data = {
            "timestamp": [b.timestamp for b in bars],
            "open": [b.open for b in bars],
            "high": [b.high for b in bars],
            "low": [b.low for b in bars],
            "close": [b.close for b in bars],
        }
        df = pd.DataFrame(data)
        df.sort_values("timestamp", inplace=True)
        df.reset_index(drop=True, inplace=True)

        self.session_manager.mark_activity()
        return df
