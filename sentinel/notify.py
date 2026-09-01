"""Discord notifications for sentinel events (same webhook as gold_monitor)."""
import logging
from datetime import datetime, timezone

import requests

logger = logging.getLogger(__name__)

COLOR_OPEN, COLOR_VETO, COLOR_CLOSE, COLOR_INFO = 0x00A86B, 0xFF8C00, 0x888888, 0x3498DB


class NullNotifier:
    def send(self, title: str, lines: list, color: int = COLOR_INFO) -> bool:
        return False


class DiscordNotifier:
    def __init__(self, webhook_url: str, bot_name: str = "Sentinel (DEMO)"):
        # the .env.example placeholder is not a webhook
        self.webhook_url = "" if "YOUR_WEBHOOK" in (webhook_url or "") else webhook_url
        self.bot_name = bot_name

    def send(self, title: str, lines: list, color: int = COLOR_INFO) -> bool:
        if not self.webhook_url:
            return False
        payload = {
            "username": self.bot_name,
            "embeds": [{
                "title": title[:256],
                "description": "\n".join(lines)[:4000],
                "color": color,
                "footer": {"text": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")},
            }],
        }
        try:
            r = requests.post(self.webhook_url, json=payload, timeout=10)
            if r.status_code in (200, 204):
                return True
            logger.error(f"Discord webhook failed: {r.status_code} {r.text[:200]}")
        except Exception as e:
            logger.error(f"Discord error: {e}")
        return False
