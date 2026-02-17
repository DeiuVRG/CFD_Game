import os
from datetime import datetime
from typing import List

from broker.models import OpenPosition
from monitoring.performance_tracker import PerformanceSummary


class Dashboard:

    @staticmethod
    def print_status(
        summary: PerformanceSummary,
        positions: List[OpenPosition],
        active_strategies: List[str],
        session_ok: bool,
        mode: str,
    ):
        os.system("cls" if os.name == "nt" else "clear")
        now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")

        day_pnl_pct = 0.0
        if summary.day_start_equity > 0:
            day_pnl_pct = (summary.day_pnl / summary.day_start_equity) * 100
        sign = "+" if summary.day_pnl >= 0 else ""

        print(f"{'='*60}")
        print(f"  CFD Trading Bot | {now} | {mode.upper()}")
        print(f"{'='*60}")
        print(
            f"  Equity: {summary.current_equity:.2f} | "
            f"Day P&L: {sign}{summary.day_pnl:.2f} ({sign}{day_pnl_pct:.1f}%)"
        )
        print(
            f"  Positions: {summary.open_positions}/3 | "
            f"Trades today: {summary.total_trades} | "
            f"Win rate: {summary.win_rate:.0f}%"
        )
        print(f"  Strategies: {', '.join(active_strategies)}")
        print(f"  Session: {'OK' if session_ok else 'DISCONNECTED'}")

        if positions:
            print(f"\n  Open Positions:")
            for p in positions:
                pnl_sign = "+" if p.profit_loss >= 0 else ""
                print(
                    f"    {p.direction:4s} {p.epic:10s} "
                    f"size={p.size} @ {p.open_level:.5f} "
                    f"P&L: {pnl_sign}{p.profit_loss:.2f}"
                )

        print(f"{'='*60}")
