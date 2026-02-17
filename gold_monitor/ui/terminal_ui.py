import os
from datetime import datetime
from typing import Optional

from engine.signal import Signal


# ANSI color codes
class C:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    WHITE = "\033[97m"
    DIM = "\033[2m"
    BG_GREEN = "\033[42m"
    BG_RED = "\033[41m"
    BG_YELLOW = "\033[43m"


class TerminalUI:

    @staticmethod
    def clear():
        os.system("cls" if os.name == "nt" else "clear")

    @staticmethod
    def display_multi(monitors, position_tracker, results: dict):
        """Display dashboard for all monitored instruments."""
        TerminalUI.clear()
        now = datetime.utcnow().strftime("%H:%M:%S UTC")
        w = 62

        print(f"{C.CYAN}{'═' * w}{C.RESET}")
        print(f"{C.BOLD}{C.CYAN}  TRADING MONITOR  │  MULTI-INSTRUMENT  │  LIVE  │  {now}{C.RESET}")
        print(f"{C.CYAN}{'═' * w}{C.RESET}")

        for mon in monitors:
            res = results.get(mon.display_name, {})
            discord_sent = res.get("discord_sent", False)
            close_sent = res.get("close_sent", False)

            # Price color
            if mon.change >= 0:
                price_color = C.GREEN
                arrow = "▲"
                sign = "+"
            else:
                price_color = C.RED
                arrow = "▼"
                sign = ""

            # Detect forex (more decimals)
            is_forex = "USD" in mon.display_name and "XAU" not in mon.display_name
            price_fmt = f"{mon.price:.5f}" if is_forex else f"${mon.price:,.2f}"
            change_fmt = f"{sign}{mon.change:.5f}" if is_forex else f"{sign}{mon.change:,.2f}"

            print(f"\n  {C.BOLD}{C.YELLOW}{mon.display_name}{C.RESET}")
            print(
                f"  {C.BOLD}{price_color}{price_fmt}{C.RESET}  "
                f"{price_color}{arrow} {change_fmt} ({sign}{mon.change_pct:.2f}%){C.RESET}"
            )

            # Indicators
            rsi_color = C.RED if mon.rsi > 70 else C.GREEN if mon.rsi < 30 else C.WHITE
            print(
                f"  RSI: {rsi_color}{mon.rsi:>5.1f}{C.RESET}  │  "
                f"MACD: {C.GREEN if mon.macd_hist > 0 else C.RED}{mon.macd_hist:>+.4f}{C.RESET}  │  "
                f"AI: {'OK' if mon.ai_ready else 'N/A'}"
            )

            # Signal
            if mon.last_signal:
                sig = mon.last_signal
                if sig.direction == "BUY":
                    sig_bg = C.BG_GREEN
                    sig_text = "CUMPARA"
                else:
                    sig_bg = C.BG_RED
                    sig_text = " VINDE "

                conf = mon.last_ai_confidence if mon.last_ai_confidence > 0 else sig.strength
                rr = sig.risk_reward_ratio

                sl_fmt = f"{sig.stop_loss:.5f}" if is_forex else f"${sig.stop_loss:,.2f}"
                tp_fmt = f"{sig.take_profit:.5f}" if is_forex else f"${sig.take_profit:,.2f}"

                print(
                    f"  {sig_bg}{C.BOLD}{C.WHITE} {sig_text} {C.RESET}  "
                    f"Conf: {conf:.0%}  │  SL: {sl_fmt}  │  TP: {tp_fmt}  │  R:R 1:{rr:.1f}"
                )
            else:
                print(f"  {C.DIM}Niciun semnal activ{C.RESET}")

            # Open position
            pos = position_tracker.get_position(mon.display_name)
            if pos:
                pnl = pos.unrealized_pnl_pct(mon.price)
                pnl_color = C.GREEN if pnl > 0 else C.RED
                pnl_sign = "+" if pnl > 0 else ""
                print(
                    f"  {C.BOLD}Pozitie deschisa:{C.RESET} {pos.direction}  │  "
                    f"P&L: {pnl_color}{pnl_sign}{pnl:.2f}%{C.RESET}"
                )

            # Discord status
            if discord_sent:
                print(f"  {C.GREEN}Discord: Semnal trimis!{C.RESET}")
            if close_sent:
                print(f"  {C.YELLOW}Discord: Pozitie inchisa!{C.RESET}")

            print(f"  {C.CYAN}{'─' * (w - 2)}{C.RESET}")

        # Stats
        stats = position_tracker.get_stats()
        if stats["total"] > 0:
            print(
                f"\n  {C.BOLD}Statistici:{C.RESET} "
                f"{stats['total']} trade-uri  │  "
                f"{C.GREEN}W: {stats['wins']}{C.RESET}  │  "
                f"{C.RED}L: {stats['losses']}{C.RESET}  │  "
                f"WR: {stats['win_rate']:.0%}  │  "
                f"P&L: {stats.get('total_pnl_pct', 0):+.2f}%"
            )

        # Data source info
        from data.gold_fetcher import credit_tracker
        src = credit_tracker.source_name
        remaining = credit_tracker.credits_remaining
        src_color = C.GREEN if src == "TwelveData" else C.YELLOW
        print(
            f"\n  {C.DIM}Sursa: {src_color}{src}{C.RESET}"
            f"{C.DIM} ({remaining} credite ramase)  │  Ctrl+C pentru a opri{C.RESET}"
        )
        print(f"{C.CYAN}{'═' * w}{C.RESET}")
