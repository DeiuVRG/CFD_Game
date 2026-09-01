"""Append-only SQLite persistence for every generated signal, plus the later
hypothetical outcome of each signal (TP/SL/EOD..., exit price, gross/net P&L).

This is the serious data-collection layer: the point of the project is to
gather EVIDENCE about signal quality before any execution ever happens.

Rules:
  - signals are append-only: rows are only inserted, core fields never change
  - the outcome columns of a row are filled exactly once (WHERE outcome IS NULL)
  - every row records the model version (sha256 of the .pkl + training date)
"""
import csv
import hashlib
import logging
import os
import sqlite3
import threading
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "signals.db")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS signals (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    ts_utc        TEXT NOT NULL,
    instrument    TEXT NOT NULL,
    direction     TEXT NOT NULL,
    confidence    REAL,
    prob_buy      REAL,
    prob_sell     REAL,
    prob_hold     REAL,
    adx           REAL,
    regime        TEXT,
    entry_price   REAL,
    stop_loss     REAL,
    take_profit   REAL,
    strategy      TEXT,
    model_version TEXT,
    -- hypothetical outcome, filled once by the position tracker
    outcome       TEXT,
    outcome_ts_utc TEXT,
    exit_price    REAL,
    pnl_gross_pct REAL,
    pnl_net_pct   REAL
);
CREATE INDEX IF NOT EXISTS idx_signals_instrument ON signals (instrument);
"""

_EXPORT_COLUMNS = [
    "id", "ts_utc", "instrument", "direction", "confidence",
    "prob_buy", "prob_sell", "prob_hold", "adx", "regime",
    "entry_price", "stop_loss", "take_profit", "strategy", "model_version",
    "outcome", "outcome_ts_utc", "exit_price", "pnl_gross_pct", "pnl_net_pct",
]


def model_version(model_path: str) -> str:
    """Version id for the model file: sha256 prefix + file date. Lets every
    signal be traced back to the exact trained model that produced it."""
    try:
        with open(model_path, "rb") as f:
            digest = hashlib.sha256(f.read()).hexdigest()[:12]
        mtime = datetime.fromtimestamp(os.path.getmtime(model_path),
                                       tz=timezone.utc)
        return f"{digest}@{mtime.strftime('%Y-%m-%d')}"
    except OSError:
        return "nomodel"


def _utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class SignalStore:
    def __init__(self, db_path: str = None):
        self.db_path = db_path or DEFAULT_DB_PATH
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.executescript(_SCHEMA)
            self._conn.commit()

    def close(self):
        with self._lock:
            self._conn.close()

    # ------------------------------------------------------------------
    # Writes
    # ------------------------------------------------------------------

    def insert_signal(
        self,
        instrument: str,
        direction: str,
        confidence: float = None,
        probabilities: dict = None,
        adx: float = None,
        regime: str = None,
        entry_price: float = None,
        stop_loss: float = None,
        take_profit: float = None,
        strategy: str = None,
        model_ver: str = None,
        ts_utc: str = None,
    ) -> int:
        """Append a new signal row; returns its id."""
        probabilities = probabilities or {}
        with self._lock:
            cur = self._conn.execute(
                """INSERT INTO signals
                   (ts_utc, instrument, direction, confidence,
                    prob_buy, prob_sell, prob_hold, adx, regime,
                    entry_price, stop_loss, take_profit, strategy, model_version)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    ts_utc or _utcnow(), instrument, direction, confidence,
                    probabilities.get("BUY"), probabilities.get("SELL"),
                    probabilities.get("HOLD"), adx, regime,
                    entry_price, stop_loss, take_profit, strategy, model_ver,
                ),
            )
            self._conn.commit()
            return int(cur.lastrowid)

    def record_outcome(
        self,
        signal_id: int,
        outcome: str,
        exit_price: float,
        pnl_gross_pct: float,
        pnl_net_pct: float,
        ts_utc: str = None,
    ) -> bool:
        """Fill the hypothetical outcome of a signal exactly once.
        Returns False when the row does not exist or already has an outcome
        (append-only discipline: outcomes are never overwritten)."""
        if signal_id is None:
            return False
        with self._lock:
            cur = self._conn.execute(
                """UPDATE signals
                   SET outcome = ?, outcome_ts_utc = ?, exit_price = ?,
                       pnl_gross_pct = ?, pnl_net_pct = ?
                   WHERE id = ? AND outcome IS NULL""",
                (outcome, ts_utc or _utcnow(), exit_price,
                 pnl_gross_pct, pnl_net_pct, signal_id),
            )
            self._conn.commit()
            updated = cur.rowcount > 0
        if not updated:
            logger.warning(
                f"Outcome for signal {signal_id} not recorded "
                f"(missing row or already finalized)"
            )
        return updated

    # ------------------------------------------------------------------
    # Reads / reporting
    # ------------------------------------------------------------------

    def fetch_all(self, instrument: str = None) -> list:
        query = "SELECT * FROM signals"
        params = ()
        if instrument:
            query += " WHERE instrument = ?"
            params = (instrument,)
        query += " ORDER BY id ASC"
        with self._lock:
            rows = self._conn.execute(query, params).fetchall()
        return [dict(r) for r in rows]

    def fetch_open(self, instrument: str = None) -> list:
        """Signals whose hypothetical outcome is still unresolved (oldest
        first). Used at startup to rebuild/replay positions after a restart."""
        query = "SELECT * FROM signals WHERE outcome IS NULL"
        params = ()
        if instrument:
            query += " AND instrument = ?"
            params = (instrument,)
        query += " ORDER BY id ASC"
        with self._lock:
            rows = self._conn.execute(query, params).fetchall()
        return [dict(r) for r in rows]

    def export_csv(self, path: str) -> int:
        """Dump every signal (with outcomes) to CSV for external audit.
        Returns the number of exported rows."""
        rows = self.fetch_all()
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=_EXPORT_COLUMNS)
            writer.writeheader()
            for row in rows:
                writer.writerow({k: row.get(k) for k in _EXPORT_COLUMNS})
        return len(rows)

    def compute_report(self) -> dict:
        """Per-instrument statistics over the hypothetical outcomes:
        signal counts, win rate, average net expectancy, and the max drawdown
        of the hypothetical equity curve (compounded net P&L in signal order).
        """
        report: dict = {}
        for row in self.fetch_all():
            inst = report.setdefault(row["instrument"], {
                "signals": 0, "closed": 0, "wins": 0,
                "pnl_net_sum": 0.0, "equity_curve": [1.0],
                "outcomes": {},
            })
            inst["signals"] += 1
            if row["outcome"] is not None:
                inst["closed"] += 1
                pnl_net = row["pnl_net_pct"] or 0.0
                if pnl_net > 0:
                    inst["wins"] += 1
                inst["pnl_net_sum"] += pnl_net
                inst["equity_curve"].append(
                    inst["equity_curve"][-1] * (1 + pnl_net / 100)
                )
                inst["outcomes"][row["outcome"]] = (
                    inst["outcomes"].get(row["outcome"], 0) + 1
                )

        for inst, stats in report.items():
            closed = stats["closed"]
            stats["win_rate"] = stats["wins"] / closed if closed else 0.0
            stats["avg_net_expectancy_pct"] = (
                stats["pnl_net_sum"] / closed if closed else 0.0
            )
            curve = stats.pop("equity_curve")
            peak = curve[0]
            max_dd = 0.0
            for v in curve:
                peak = max(peak, v)
                max_dd = min(max_dd, (v - peak) / peak)
            stats["hypothetical_return_pct"] = (curve[-1] - 1) * 100
            stats["max_drawdown_pct"] = max_dd * 100

        return report
