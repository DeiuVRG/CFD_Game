import time
import logging

from broker.capital_client import CapitalClient, CapitalAPIError
from config.credentials import Credentials
from config.settings import BROKER

logger = logging.getLogger(__name__)


class SessionManager:
    def __init__(self, client: CapitalClient, credentials: Credentials):
        self.client = client
        self.credentials = credentials
        self._last_activity: float = 0.0
        self._login_count: int = 0

    @property
    def is_active(self) -> bool:
        return self.client.is_authenticated

    def login(self) -> bool:
        for attempt in range(BROKER.MAX_RETRIES):
            try:
                self.client.create_session(
                    api_key=self.credentials.api_key,
                    identifier=self.credentials.identifier,
                    password=self.credentials.password,
                )
                self._last_activity = time.time()
                self._login_count += 1
                logger.info(
                    f"Logged in to Capital.com ({self.credentials.mode}) "
                    f"[login #{self._login_count}]"
                )
                return True
            except CapitalAPIError as e:
                logger.error(f"Login attempt {attempt + 1} failed: {e}")
                if attempt < BROKER.MAX_RETRIES - 1:
                    time.sleep(BROKER.RETRY_DELAY_SEC * (attempt + 1))
        return False

    def ensure_session(self) -> bool:
        if not self.client.is_authenticated:
            return self.login()

        elapsed = time.time() - self._last_activity
        if elapsed >= BROKER.SESSION_REFRESH_SEC:
            if self.client.ping():
                self._last_activity = time.time()
                logger.debug("Session refreshed via ping")
                return True
            else:
                logger.warning("Ping failed, re-logging in...")
                return self.login()

        return True

    def mark_activity(self):
        self._last_activity = time.time()

    def logout(self):
        self.client.delete_session()
        logger.info("Logged out from Capital.com")
