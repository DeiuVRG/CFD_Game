"""Capital.com DEMO adapter over execution_capital's client.

Imports execution_capital lazily (its packages are named config/data/engine
like gold_monitor's - never load both in one process). Refuses anything but
CAPITAL_MODE=demo: live is Gate 3, a separate explicit decision.
"""
import logging
import os
import sys
from typing import Optional

import pandas as pd

from sentinel.config import EXECUTION_CAPITAL_DIR, SentinelConfig

logger = logging.getLogger(__name__)


class DemoBroker:
    def __init__(self, cfg: SentinelConfig):
        if cfg.capital_mode != "demo":
            raise RuntimeError(
                "The sentinel runs on the DEMO account only (CAPITAL_MODE=demo). "
                "Live execution is Gate 3 - not a config switch.")
        if EXECUTION_CAPITAL_DIR not in sys.path:
            sys.path.insert(0, EXECUTION_CAPITAL_DIR)
        from broker.capital_client import CapitalAPIError, CapitalClient
        from broker.session_manager import SessionManager
        from config.credentials import Credentials
        from config.settings import BROKER

        api_key = os.getenv("CAPITAL_API_KEY", "")
        identifier = os.getenv("CAPITAL_IDENTIFIER", "")
        password = os.getenv("CAPITAL_PASSWORD", "")
        missing = [n for n, v in (("CAPITAL_API_KEY", api_key),
                                  ("CAPITAL_IDENTIFIER", identifier),
                                  ("CAPITAL_PASSWORD", password)) if not v]
        if missing:
            raise RuntimeError(f"Missing Capital.com demo credentials: {', '.join(missing)}")

        self._api_error = CapitalAPIError
        self.client = CapitalClient(BROKER.DEMO_URL)
        self.session = SessionManager(self.client, Credentials(
            api_key=api_key, identifier=identifier, password=password,
            mode="demo", base_url=BROKER.DEMO_URL))
        if not self.session.login():
            raise RuntimeError("Capital.com demo login failed")
        self._market_info = {}

    # ---- account -----------------------------------------------------
    def equity(self) -> float:
        self.session.ensure_session()
        return float(self.client.get_account_equity())

    def positions(self) -> list:
        self.session.ensure_session()
        out = []
        for p in self.client.get_positions():
            out.append({
                "deal_id": p.deal_id, "epic": p.epic, "direction": p.direction,
                "size": p.size, "open_level": p.open_level,
                "stop_level": float(p.stop_level) if p.stop_level is not None else None,
                "profit_level": float(p.profit_level) if p.profit_level is not None else None,
                "profit_loss": p.profit_loss, "created": p.created_date,
            })
        return out

    def market_info(self, epic: str) -> Optional[dict]:
        self.session.ensure_session()
        info = self.client.get_market_info(epic)
        if info is None:
            return None
        return {"epic": info.epic, "name": info.instrument_name,
                "min_size": info.min_deal_size, "max_size": info.max_deal_size,
                "pip_size": info.pip_size, "market_open": info.market_open}

    def candles(self, epic: str, resolution: str = "HOUR", count: int = 100) -> pd.DataFrame:
        self.session.ensure_session()
        bars = self.client.get_prices(epic, resolution, max_bars=count)
        if not bars:
            return pd.DataFrame()
        df = pd.DataFrame({
            "timestamp": [b.timestamp for b in bars], "open": [b.open for b in bars],
            "high": [b.high for b in bars], "low": [b.low for b in bars],
            "close": [b.close for b in bars],
        })
        return df.sort_values("timestamp").reset_index(drop=True)

    # ---- orders ------------------------------------------------------
    def open(self, epic: str, direction: str, size: float,
             stop_loss: float, take_profit: float) -> dict:
        self.session.ensure_session()
        try:
            c = self.client.create_position(epic=epic, direction=direction, size=size,
                                            stop_loss=stop_loss, take_profit=take_profit)
            return {"status": c.status, "deal_id": c.deal_id, "level": c.level,
                    "reason": c.reason}
        except self._api_error as e:
            logger.error(f"create_position failed: {e}")
            return {"status": "ERROR", "deal_id": "", "level": 0, "reason": str(e)}

    def close(self, deal_id: str) -> dict:
        self.session.ensure_session()
        try:
            c = self.client.close_position(deal_id)
            return {"status": c.status, "level": c.level, "profit": c.profit, "reason": c.reason}
        except self._api_error as e:
            logger.error(f"close_position failed: {e}")
            return {"status": "ERROR", "reason": str(e)}

    def tighten_sl(self, deal_id: str, new_stop_loss: float) -> dict:
        self.session.ensure_session()
        try:
            return {"status": "OK", "response": self.client.update_position(
                deal_id, stop_level=new_stop_loss)}
        except self._api_error as e:
            logger.error(f"update_position failed: {e}")
            return {"status": "ERROR", "reason": str(e)}

    def search(self, term: str) -> list:
        self.session.ensure_session()
        return self.client.search_markets(term)
