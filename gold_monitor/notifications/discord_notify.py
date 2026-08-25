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
        self._last_signal_direction: dict[str, str] = {}

    def _post(self, payload: dict) -> bool:
        if not self.webhook_url:
            logger.warning("Discord webhook URL not configured")
            return False

        # No mention configured (DISCORD_MENTION unset) -> drop empty content
        if not payload.get("content"):
            payload.pop("content", None)

        try:
            resp = requests.post(self.webhook_url, json=payload, timeout=10)
            if resp.status_code in (200, 204):
                return True
            logger.error(f"Discord webhook failed: {resp.status_code} {resp.text[:200]}")
            return False
        except Exception as e:
            logger.error(f"Discord notification error: {e}")
            return False

    def send_signal(self, signal: Signal, confidence: float = 0,
                    probabilities: dict = None) -> bool:
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

        # 5-decimal format only for sub-10 prices (FX pairs); gold/BTC
        # and other large prices use thousands formatting.
        fmt = "5f" if signal.entry_price < 10 else ",.2f"

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
            "footer": {"text": f"\u23f0 {now} | Trailing SL activ"},
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
        reason_map = {
            "TP_HIT": ("\U0001f3af TARGET ATINS!", 0x00FF00, "Profit realizat!"),
            "SL_HIT": ("\U0001f6d1 STOP LOSS ATINS!", 0xFF0000, "Pierdere limitata."),
            "TRAILING_SL_HIT": ("\U0001f4c8 TRAILING SL ATINS!", 0xFFAA00, "Profit protejat prin trailing stop!"),
            "SIGNAL_REVERSED": ("\U0001f504 SEMNAL INVERSAT!", 0xFFAA00, "Semnalul s-a schimbat."),
            "EOD_CLOSE": ("\U0001f319 INCHIDERE END-OF-DAY!", 0x888888, "Pozitia nu ramane peste noapte."),
            "MANUAL_WIN": ("\u2705 INCHIS MANUAL - PROFIT!", 0x00FF00, "Inchis manual pe profit."),
            "MANUAL_LOSS": ("\u274c INCHIS MANUAL - PIERDERE!", 0xFF0000, "Inchis manual pe pierdere."),
        }

        title, color, desc = reason_map.get(
            position.close_reason,
            ("POZITIE INCHISA", 0x888888, ""),
        )

        pnl = position.pnl_pct
        pnl_emoji = "\U0001f4b0" if pnl > 0 else "\U0001f4b8"
        pnl_sign = "+" if pnl > 0 else ""

        fmt = "5f" if position.entry_price < 10 else ",.2f"

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

    def send_test(self) -> bool:
        from config.settings import INSTRUMENTS
        instruments_str = ", ".join(i.SYMBOL_DISPLAY for i in INSTRUMENTS if i.ENABLED)

        embed = {
            "title": "\u2705 Trading Monitor v2 - Test Message",
            "description": (
                f"Monitorizare activa: {instruments_str}\n"
                f"Features: Walk-forward AI, Trailing SL, Session filter, Regime detection"
            ),
            "color": 0x00AAFF,
            "fields": [
                {"name": "Status", "value": "Conectat", "inline": True},
                {"name": "Versiune", "value": "v2.0", "inline": True},
            ],
            "footer": {"text": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")},
        }
        return self._post({"username": DISCORD.BOT_NAME, "content": DISCORD.MENTION_TAG, "embeds": [embed]})
