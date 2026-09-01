"""Append-only SQLite log of every sentinel decision + research briefs +
small key/value state (signal cursor, day-start equity)."""
import json
import os
import sqlite3
import threading
from datetime import datetime, timezone
from typing import Optional

_SCHEMA = """
CREATE TABLE IF NOT EXISTS decisions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts_utc TEXT NOT NULL,
    kind TEXT NOT NULL,                 -- OPEN | REVIEW
    instrument TEXT, epic TEXT, signal_id INTEGER, deal_id TEXT,
    direction TEXT, entry_price REAL, stop_loss REAL, take_profit REAL, size REAL,
    llm_action TEXT, llm_size_fraction REAL, llm_confidence REAL,
    llm_rationale TEXT, llm_risks TEXT,
    final_action TEXT NOT NULL,         -- OPEN | DRY_RUN | VETO | REJECT | SKIP | HOLD | CLOSE | TIGHTEN_SL
    reason TEXT,
    model TEXT, input_tokens INTEGER, output_tokens INTEGER, research_id INTEGER,
    dry_run INTEGER NOT NULL DEFAULT 0,
    last_pnl REAL,
    outcome TEXT, outcome_ts_utc TEXT, pnl REAL
);
CREATE TABLE IF NOT EXISTS research (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts_utc TEXT NOT NULL, instrument TEXT NOT NULL, brief TEXT,
    model TEXT, input_tokens INTEGER, output_tokens INTEGER
);
CREATE TABLE IF NOT EXISTS state (key TEXT PRIMARY KEY, value TEXT);
"""


def utcnow_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class DecisionStore:
    def __init__(self, db_path: str):
        os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.executescript(_SCHEMA)
            self._conn.commit()

    def close(self):
        with self._lock:
            self._conn.close()

    # ---- state -----------------------------------------------------------
    def get_state(self, key: str, default=None):
        with self._lock:
            row = self._conn.execute("SELECT value FROM state WHERE key = ?", (key,)).fetchone()
        return row["value"] if row else default

    def set_state(self, key: str, value):
        with self._lock:
            self._conn.execute(
                "INSERT INTO state (key, value) VALUES (?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (key, str(value)))
            self._conn.commit()

    # ---- research --------------------------------------------------------
    def insert_research(self, instrument: str, brief: str, model: str = None,
                        input_tokens: int = None, output_tokens: int = None,
                        ts_utc: str = None) -> int:
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO research (ts_utc, instrument, brief, model, input_tokens, output_tokens) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (ts_utc or utcnow_iso(), instrument, brief, model, input_tokens, output_tokens))
            self._conn.commit()
            return int(cur.lastrowid)

    def latest_research(self, instrument: str, max_age_sec: int, now: datetime = None) -> Optional[dict]:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM research WHERE instrument = ? ORDER BY id DESC LIMIT 1",
                (instrument,)).fetchone()
        if row is None:
            return None
        now = now or datetime.now(timezone.utc)
        ts = datetime.fromisoformat(row["ts_utc"].replace("Z", "+00:00"))
        if (now - ts).total_seconds() > max_age_sec:
            return None
        return dict(row)

    # ---- decisions -------------------------------------------------------
    def insert_decision(self, **f) -> int:
        f.setdefault("ts_utc", utcnow_iso())
        if isinstance(f.get("llm_risks"), (list, tuple)):
            f["llm_risks"] = json.dumps(list(f["llm_risks"]), ensure_ascii=False)
        f["dry_run"] = 1 if f.get("dry_run") else 0
        cols = ", ".join(f.keys())
        marks = ", ".join("?" for _ in f)
        with self._lock:
            cur = self._conn.execute(
                f"INSERT INTO decisions ({cols}) VALUES ({marks})", tuple(f.values()))
            self._conn.commit()
            return int(cur.lastrowid)

    def open_trades(self) -> list:
        """Executed demo trades whose outcome is not known yet."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM decisions WHERE kind = 'OPEN' AND final_action = 'OPEN' "
                "AND deal_id IS NOT NULL AND outcome IS NULL ORDER BY id ASC").fetchall()
        return [dict(r) for r in rows]

    def update_last_pnl(self, decision_id: int, pnl: float):
        with self._lock:
            self._conn.execute("UPDATE decisions SET last_pnl = ? WHERE id = ?",
                               (pnl, decision_id))
            self._conn.commit()

    def record_outcome(self, decision_id: int, outcome: str, pnl: float = None,
                       ts_utc: str = None) -> bool:
        with self._lock:
            cur = self._conn.execute(
                "UPDATE decisions SET outcome = ?, outcome_ts_utc = ?, pnl = ? "
                "WHERE id = ? AND outcome IS NULL",
                (outcome, ts_utc or utcnow_iso(), pnl, decision_id))
            self._conn.commit()
            return cur.rowcount > 0

    def trades_today(self, day: str = None) -> int:
        day = day or utcnow_iso()[:10]
        with self._lock:
            row = self._conn.execute(
                "SELECT COUNT(*) AS n FROM decisions WHERE kind = 'OPEN' "
                "AND final_action = 'OPEN' AND ts_utc LIKE ?", (day + "%",)).fetchone()
        return int(row["n"])

    def fetch_all(self, kind: str = None) -> list:
        q, p = "SELECT * FROM decisions", ()
        if kind:
            q, p = q + " WHERE kind = ?", (kind,)
        with self._lock:
            rows = self._conn.execute(q + " ORDER BY id ASC", p).fetchall()
        return [dict(r) for r in rows]
