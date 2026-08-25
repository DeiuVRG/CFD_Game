import time
import logging
import requests
from typing import Optional, List, Dict, Any

from broker.models import (
    PriceBar, MarketInfo, OpenPosition, TradeConfirmation, AccountInfo,
)
from config.settings import BROKER

logger = logging.getLogger(__name__)


class CapitalAPIError(Exception):
    def __init__(self, status_code: int, message: str):
        self.status_code = status_code
        self.message = message
        super().__init__(f"[{status_code}] {message}")


class CapitalClient:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
        self.cst: Optional[str] = None
        self.security_token: Optional[str] = None
        self._last_request_time: float = 0.0
        self._last_order_time: float = 0.0

    @property
    def is_authenticated(self) -> bool:
        return self.cst is not None and self.security_token is not None

    def _rate_limit(self, is_order: bool = False):
        now = time.time()
        if is_order:
            elapsed = now - self._last_order_time
            if elapsed < BROKER.ORDER_RATE_LIMIT_SEC:
                time.sleep(BROKER.ORDER_RATE_LIMIT_SEC - elapsed)
            self._last_order_time = time.time()
        else:
            elapsed = now - self._last_request_time
            if elapsed < BROKER.RATE_LIMIT_SEC:
                time.sleep(BROKER.RATE_LIMIT_SEC - elapsed)
            self._last_request_time = time.time()

    def _auth_headers(self) -> Dict[str, str]:
        return {
            "CST": self.cst or "",
            "X-SECURITY-TOKEN": self.security_token or "",
            "Content-Type": "application/json",
        }

    def _request(
        self, method: str, path: str,
        json_body: Optional[Dict] = None,
        params: Optional[Dict] = None,
        headers: Optional[Dict] = None,
        is_order: bool = False,
    ) -> requests.Response:
        self._rate_limit(is_order)
        url = f"{self.base_url}{path}"
        hdrs = headers or self._auth_headers()

        for attempt in range(BROKER.MAX_RETRIES):
            try:
                resp = self.session.request(
                    method, url, json=json_body, params=params,
                    headers=hdrs, timeout=15,
                )
                if resp.status_code in (200, 201):
                    return resp
                if resp.status_code == 401:
                    raise CapitalAPIError(401, "Session expired or invalid credentials")
                if resp.status_code == 429:
                    wait = BROKER.RETRY_DELAY_SEC * (attempt + 1)
                    logger.warning(f"Rate limited, waiting {wait}s...")
                    time.sleep(wait)
                    continue
                error_msg = resp.text[:500]
                raise CapitalAPIError(resp.status_code, error_msg)
            except requests.exceptions.RequestException as e:
                if attempt < BROKER.MAX_RETRIES - 1:
                    logger.warning(f"Request failed (attempt {attempt + 1}): {e}")
                    time.sleep(BROKER.RETRY_DELAY_SEC)
                else:
                    raise CapitalAPIError(0, f"Connection failed: {e}")

    # --- Session ---

    def create_session(self, api_key: str, identifier: str, password: str) -> bool:
        headers = {
            "X-CAP-API-KEY": api_key,
            "Content-Type": "application/json",
        }
        body = {
            "identifier": identifier,
            "password": password,
            "encryptedPassword": False,
        }
        resp = self._request("POST", "/api/v1/session", json_body=body, headers=headers)
        self.cst = resp.headers.get("CST")
        self.security_token = resp.headers.get("X-SECURITY-TOKEN")

        if not self.cst or not self.security_token:
            raise CapitalAPIError(0, "No CST or X-SECURITY-TOKEN in response headers")

        logger.info("Session created successfully")
        return True

    def ping(self) -> bool:
        try:
            self._request("GET", "/api/v1/ping")
            return True
        except CapitalAPIError:
            return False

    def delete_session(self):
        try:
            self._request("DELETE", "/api/v1/session")
            logger.info("Session deleted")
        except CapitalAPIError as e:
            logger.warning(f"Failed to delete session: {e}")
        finally:
            self.cst = None
            self.security_token = None

    # --- Account ---

    def get_accounts(self) -> List[AccountInfo]:
        resp = self._request("GET", "/api/v1/accounts")
        data = resp.json()
        accounts = []
        for acc in data.get("accounts", []):
            accounts.append(AccountInfo(
                account_id=acc.get("accountId", ""),
                balance=float(acc.get("balance", {}).get("balance", 0)),
                deposit=float(acc.get("balance", {}).get("deposit", 0)),
                profit_loss=float(acc.get("balance", {}).get("profitLoss", 0)),
                available=float(acc.get("balance", {}).get("available", 0)),
                currency=acc.get("currency", "EUR"),
            ))
        return accounts

    def get_account_equity(self) -> float:
        accounts = self.get_accounts()
        if not accounts:
            return 0.0
        acc = accounts[0]
        return acc.balance + acc.profit_loss

    # --- Positions ---

    def get_positions(self) -> List[OpenPosition]:
        resp = self._request("GET", "/api/v1/positions")
        data = resp.json()
        positions = []
        for item in data.get("positions", []):
            pos = item.get("position", {})
            market = item.get("market", {})
            positions.append(OpenPosition(
                deal_id=pos.get("dealId", ""),
                epic=market.get("epic", pos.get("epic", "")),
                direction=pos.get("direction", ""),
                size=float(pos.get("size", 0)),
                open_level=float(pos.get("level", 0)),
                stop_level=pos.get("stopLevel"),
                profit_level=pos.get("profitLevel"),
                profit_loss=float(pos.get("profit", 0)),
                created_date=pos.get("createdDateUTC", ""),
            ))
        return positions

    def create_position(
        self, epic: str, direction: str, size: float,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        guaranteed_stop: bool = False,
    ) -> TradeConfirmation:
        body: Dict[str, Any] = {
            "epic": epic,
            "direction": direction,
            "size": size,
            "guaranteedStop": guaranteed_stop,
        }
        if stop_loss is not None:
            if direction == "BUY":
                body["stopLevel"] = round(stop_loss, 5)
                body["stopDistance"] = None
            else:
                body["stopLevel"] = round(stop_loss, 5)
                body["stopDistance"] = None
        if take_profit is not None:
            body["profitLevel"] = round(take_profit, 5)

        resp = self._request("POST", "/api/v1/positions", json_body=body, is_order=True)
        data = resp.json()
        deal_ref = data.get("dealReference", "")

        if deal_ref:
            return self.get_deal_confirmation(deal_ref)

        return TradeConfirmation(
            deal_id="",
            deal_reference=deal_ref,
            status="UNKNOWN",
            reason=str(data),
            level=0, size=size, direction=direction, profit=0,
        )

    def close_position(self, deal_id: str) -> TradeConfirmation:
        resp = self._request("DELETE", f"/api/v1/positions/{deal_id}", is_order=True)
        data = resp.json()
        deal_ref = data.get("dealReference", "")
        if deal_ref:
            return self.get_deal_confirmation(deal_ref)
        return TradeConfirmation(
            deal_id=deal_id, deal_reference=deal_ref,
            status="CLOSED", reason="", level=0, size=0,
            direction="", profit=0,
        )

    def update_position(
        self, deal_id: str,
        stop_level: Optional[float] = None,
        profit_level: Optional[float] = None,
    ) -> Dict:
        body: Dict[str, Any] = {}
        if stop_level is not None:
            body["stopLevel"] = round(stop_level, 5)
        if profit_level is not None:
            body["profitLevel"] = round(profit_level, 5)
        resp = self._request("PUT", f"/api/v1/positions/{deal_id}", json_body=body, is_order=True)
        return resp.json()

    def get_deal_confirmation(self, deal_reference: str) -> TradeConfirmation:
        time.sleep(0.5)  # Brief wait for broker to process
        resp = self._request("GET", f"/api/v1/confirms/{deal_reference}")
        data = resp.json()
        return TradeConfirmation(
            deal_id=data.get("dealId", ""),
            deal_reference=deal_reference,
            status=data.get("dealStatus", "UNKNOWN"),
            reason=data.get("reason", ""),
            level=float(data.get("level", 0)),
            size=float(data.get("size", 0)),
            direction=data.get("direction", ""),
            profit=float(data.get("profit", 0)),
            affected_deals=data.get("affectedDeals", []),
        )

    # --- Market Data ---

    def get_prices(
        self, epic: str, resolution: str, max_bars: int = 100,
    ) -> List[PriceBar]:
        params = {
            "resolution": resolution,
            "max": max_bars,
        }
        resp = self._request("GET", f"/api/v1/prices/{epic}", params=params)
        data = resp.json()

        bars = []
        for candle in data.get("prices", []):
            snap_time = candle.get("snapshotTimeUTC", candle.get("snapshotTime", ""))
            bid = candle.get("closePrice", candle.get("bid", {}))
            ask = candle.get("openPrice", candle.get("ask", {}))

            # Capital.com returns bid/ask separately; use mid prices
            if isinstance(bid, dict):
                o = (float(candle.get("openPrice", {}).get("bid", 0)) +
                     float(candle.get("openPrice", {}).get("ask", 0))) / 2
                h = (float(candle.get("highPrice", {}).get("bid", 0)) +
                     float(candle.get("highPrice", {}).get("ask", 0))) / 2
                l = (float(candle.get("lowPrice", {}).get("bid", 0)) +
                     float(candle.get("lowPrice", {}).get("ask", 0))) / 2
                c = (float(candle.get("closePrice", {}).get("bid", 0)) +
                     float(candle.get("closePrice", {}).get("ask", 0))) / 2
            else:
                o = float(candle.get("openPrice", 0))
                h = float(candle.get("highPrice", 0))
                l = float(candle.get("lowPrice", 0))
                c = float(candle.get("closePrice", 0))

            from datetime import datetime
            try:
                ts = datetime.fromisoformat(snap_time.replace("Z", "+00:00"))
            except (ValueError, AttributeError):
                ts = datetime.now()

            bars.append(PriceBar(timestamp=ts, open=o, high=h, low=l, close=c))

        return bars

    def get_market_info(self, epic: str) -> Optional[MarketInfo]:
        resp = self._request("GET", f"/api/v1/markets/{epic}")
        data = resp.json()
        inst = data.get("instrument", {})
        snap = data.get("snapshot", {})
        dr = data.get("dealingRules", {})

        min_size = float(
            dr.get("minDealSize", {}).get("value", 0.01)
        )
        max_size = float(
            dr.get("maxDealSize", {}).get("value", 1000)
        )

        return MarketInfo(
            epic=epic,
            instrument_name=inst.get("name", epic),
            min_deal_size=min_size,
            max_deal_size=max_size,
            pip_size=float(inst.get("pipSize", 0.0001)),
            margin_factor=float(inst.get("marginFactor", 3.33)),
            market_open=snap.get("marketStatus", "") == "TRADEABLE",
        )

    def search_markets(self, search_term: str) -> List[Dict]:
        resp = self._request(
            "GET", "/api/v1/markets",
            params={"searchTerm": search_term},
        )
        return resp.json().get("markets", [])
