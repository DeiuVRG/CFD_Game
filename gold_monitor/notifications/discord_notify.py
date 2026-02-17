import logging
from datetime import datetime

import requests

from config.settings import DISCORD
from engine.signal import Signal
from engine.position_tracker import TrackedPosition

logger = logging.getLogger(__name__)


class DiscordNotifier:
    """Send trading signals to Discord via webhook."""

    def __init__(self):
        self.webhook_url = DISCORD.WEBHOOK_URL
        self._last_signal_direction: dict[str, str] = {}  # per instrument

    def _post(self, payload: dict) -> bool:
        if not self.webhook_url:
            logger.warning("Discord webhook URL not configured")
            return False

        try:
            resp = requests.post(
                self.webhook_url,
                json=payload,
                timeout=10,
            )
            if resp.status_code in (200, 204):
                return True
            logger.error(f"Discord webhook failed: {resp.status_code} {resp.text[:200]}")
            return False
        except Exception as e:
            logger.error(f"Discord notification error: {e}")
            return False

    def send_signal(self, signal: Signal, confidence: float = 0,
                    probabilities: dict = None) -> bool:
        """Send a BUY/SELL signal to Discord with a rich embed."""
        instrument_key = signal.epic

        if DISCORD.NOTIFY_ON_SIGNAL_CHANGE:
            if signal.direction == self._last_signal_direction.get(instrument_key):
                return False
            self._last_signal_direction[instrument_key] = signal.direction

        is_buy = signal.direction == "BUY"
        color = 0x00FF00 if is_buy else 0xFF0000
        emoji = "\U0001f7e2" if is_buy else "\U0001f534"
        action = "CUMPARA" if is_buy else "VINDE"

        rr = signal.risk_reward_ratio
        now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")

        # Format prices based on instrument type
        is_forex = "USD" in signal.epic and "XAU" not in signal.epic
        fmt = "5f" if is_forex else ",.2f"

        fields = [
            {"name": "\U0001f4b0 Pret", "value": f"${signal.entry_price:{fmt}}", "inline": True},
            {"name": "\U0001f3af Target", "value": f"${signal.take_profit:{fmt}}", "inline": True},
            {"name": "\U0001f6d1 Stop Loss", "value": f"${signal.stop_loss:{fmt}}", "inline": True},
            {"name": "\U0001f4ca Strategie", "value": signal.strategy_name, "inline": True},
            {"name": "\U0001f4aa Confidenta", "value": f"{confidence:.0%}" if confidence else f"{signal.strength:.0%}", "inline": True},
            {"name": "\u2696\ufe0f R:R", "value": f"1:{rr:.1f}", "inline": True},
        ]

        if probabilities:
            prob_str = " | ".join(f"{k}: {v:.0%}" for k, v in probabilities.items())
            fields.append({"name": "\U0001f9e0 AI Probabilities", "value": prob_str, "inline": False})

        embed = {
            "title": f"{emoji} {action} {signal.epic}",
            "color": color,
            "fields": fields,
            "footer": {"text": f"\u23f0 {now}"},
        }

        payload = {
            "username": DISCORD.BOT_NAME,
            "content": DISCORD.MENTION_TAG,
            "embeds": [embed],
        }

        success = self._post(payload)
        if success:
            logger.info(f"Discord: {signal.direction} signal sent for {signal.epic}")
        return success

    def send_close_signal(self, position: TrackedPosition) -> bool:
        """Send a position CLOSE notification to Discord."""
        reason_map = {
            "TP_HIT": ("\U0001f3af TARGET ATINS!", 0x00FF00, "Profit realizat!"),
            "SL_HIT": ("\U0001f6d1 STOP LOSS ATINS!", 0xFF0000, "Pierdere limitata."),
            "SIGNAL_REVERSED": ("\U0001f504 SEMNAL INVERSAT!", 0xFFAA00, "Semnalul s-a schimbat."),
        }

        title, color, desc = reason_map.get(
            position.close_reason,
            ("POZITIE INCHISA", 0x888888, ""),
        )

        pnl = position.pnl_pct
        pnl_emoji = "\U0001f4b0" if pnl > 0 else "\U0001f4b8"
        pnl_sign = "+" if pnl > 0 else ""

        is_forex = "USD" in position.instrument and "XAU" not in position.instrument
        fmt = "5f" if is_forex else ",.2f"

        duration = ""
        if position.closed_at and position.opened_at:
            delta = position.closed_at - position.opened_at
            mins = int(delta.total_seconds() / 60)
            if mins >= 60:
                duration = f"{mins // 60}h {mins % 60}m"
            else:
                duration = f"{mins}m"

        fields = [
            {"name": "\U0001f4c8 Directie", "value": position.direction, "inline": True},
            {"name": "\U0001f4b0 Intrare", "value": f"${position.entry_price:{fmt}}", "inline": True},
            {"name": "\U0001f3f7\ufe0f Iesire", "value": f"${position.close_price:{fmt}}", "inline": True},
            {"name": f"{pnl_emoji} P&L", "value": f"{pnl_sign}{pnl:.2f}%", "inline": True},
            {"name": "\u23f1\ufe0f Durata", "value": duration or "N/A", "inline": True},
            {"name": "\U0001f4ca Strategie", "value": position.strategy_name, "inline": True},
        ]

        now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
        embed = {
            "title": f"{title} - {position.instrument}",
            "description": desc,
            "color": color,
            "fields": fields,
            "footer": {"text": f"\u23f0 {now}"},
        }

        payload = {
            "username": DISCORD.BOT_NAME,
            "content": DISCORD.MENTION_TAG,
            "embeds": [embed],
        }

        success = self._post(payload)
        if success:
            logger.info(
                f"Discord: CLOSE sent for {position.instrument} "
                f"reason={position.close_reason} pnl={pnl:+.2f}%"
            )
        return success

    def send_status(self, price: float, change: float, change_pct: float,
                    instrument: str = "XAU/USD") -> bool:
        """Send a periodic price status update."""
        emoji = "\u2b06\ufe0f" if change >= 0 else "\u2b07\ufe0f"
        sign = "+" if change >= 0 else ""
        color = 0x00AA00 if change >= 0 else 0xAA0000

        embed = {
            "title": f"\U0001f4c8 {instrument} Status Update",
            "color": color,
            "fields": [
                {"name": "Pret", "value": f"${price:,.2f}", "inline": True},
                {"name": "Schimbare", "value": f"{emoji} {sign}{change:,.2f} ({sign}{change_pct:.2f}%)", "inline": True},
            ],
            "footer": {"text": datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")},
        }

        return self._post({"username": DISCORD.BOT_NAME, "content": DISCORD.MENTION_TAG, "embeds": [embed]})

    def send_test(self) -> bool:
        """Send a test message to verify webhook works."""
        from config.settings import INSTRUMENTS
        instruments_str = ", ".join(i.SYMBOL_DISPLAY for i in INSTRUMENTS if i.ENABLED)

        embed = {
            "title": "\u2705 Trading Monitor - Test Message",
            "description": f"Monitorizare activa pentru: {instruments_str}",
            "color": 0x00AAFF,
            "fields": [
                {"name": "Status", "value": "Conectat", "inline": True},
                {"name": "Instrumente", "value": str(sum(1 for i in INSTRUMENTS if i.ENABLED)), "inline": True},
                {"name": "Mod", "value": "Test", "inline": True},
            ],
            "footer": {"text": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")},
        }
        return self._post({"username": DISCORD.BOT_NAME, "content": DISCORD.MENTION_TAG, "embeds": [embed]})
