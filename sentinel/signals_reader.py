"""Read-only access to gold_monitor's signals.db.

Deliberately NOT importing gold_monitor: the sentinel process also loads
execution_capital's broker packages, and the two apps define clashing
top-level package names (config/data/engine)."""
import os
import sqlite3
from typing import Optional


class SignalsReader:
    def __init__(self, db_path: str):
        self.db_path = db_path

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True, timeout=5)
        conn.row_factory = sqlite3.Row
        return conn

    @property
    def exists(self) -> bool:
        return os.path.exists(self.db_path)

    def fetch_since(self, last_id: int) -> list:
        if not self.exists:
            return []
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM signals WHERE id > ? ORDER BY id ASC", (int(last_id),)
            ).fetchall()
        return [dict(r) for r in rows]

    def latest_id(self) -> int:
        if not self.exists:
            return 0
        with self._conn() as conn:
            row = conn.execute("SELECT MAX(id) AS m FROM signals").fetchone()
        return int(row["m"] or 0)

    def stats(self, instrument: str, last_n: int = 200) -> dict:
        """Hypothetical-outcome statistics of the deterministic path for one
        instrument (what the model gets as 'track record')."""
        empty = {"signals": 0, "closed": 0, "win_rate": 0.0,
                 "avg_net_pct": 0.0, "last_outcomes": []}
        if not self.exists:
            return empty
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT direction, outcome, pnl_net_pct, ts_utc FROM signals "
                "WHERE instrument = ? ORDER BY id DESC LIMIT ?",
                (instrument, last_n)).fetchall()
        rows = [dict(r) for r in rows]
        closed = [r for r in rows if r["outcome"] is not None]
        wins = [r for r in closed if (r["pnl_net_pct"] or 0) > 0]
        return {
            "signals": len(rows),
            "closed": len(closed),
            "win_rate": len(wins) / len(closed) if closed else 0.0,
            "avg_net_pct": (sum(r["pnl_net_pct"] or 0 for r in closed) / len(closed)
                            if closed else 0.0),
            "last_outcomes": [f"{r['direction']}:{r['outcome']}:{(r['pnl_net_pct'] or 0):+.2f}%"
                              for r in closed[:8]],
        }
