"""Tests for the append-only signal store, the outcome pipeline through
PositionTracker, and the per-instrument report aggregation."""
import os

import pytest

from data.signal_store import SignalStore, model_version
from engine.position_tracker import PositionTracker


@pytest.fixture
def store(tmp_path):
    s = SignalStore(db_path=str(tmp_path / "signals.db"))
    yield s
    s.close()


def insert_sample(store, instrument="XAU/USD (Gold)", direction="BUY"):
    return store.insert_signal(
        instrument=instrument, direction=direction,
        confidence=0.72,
        probabilities={"BUY": 0.72, "SELL": 0.10, "HOLD": 0.18},
        adx=31.5, regime="TRENDING",
        entry_price=3000.0, stop_loss=2980.0, take_profit=3040.0,
        strategy="AI (72%)", model_ver="abc123@2026-08-25",
    )


def test_insert_and_fetch_roundtrip(store):
    sid = insert_sample(store)
    rows = store.fetch_all()
    assert len(rows) == 1
    row = rows[0]
    assert row["id"] == sid
    assert row["instrument"] == "XAU/USD (Gold)"
    assert row["direction"] == "BUY"
    assert row["prob_buy"] == pytest.approx(0.72)
    assert row["adx"] == pytest.approx(31.5)
    assert row["regime"] == "TRENDING"
    assert row["model_version"] == "abc123@2026-08-25"
    assert row["outcome"] is None


def test_outcome_recorded_exactly_once(store):
    sid = insert_sample(store)
    ok = store.record_outcome(sid, "TP_HIT", 3040.0, 1.333, 1.30)
    assert ok is True
    # Second attempt must be refused (append-only outcomes)
    ok2 = store.record_outcome(sid, "SL_HIT", 2980.0, -0.667, -0.70)
    assert ok2 is False

    row = store.fetch_all()[0]
    assert row["outcome"] == "TP_HIT"
    assert row["exit_price"] == pytest.approx(3040.0)
    assert row["pnl_net_pct"] == pytest.approx(1.30)


def test_outcome_for_unknown_signal_refused(store):
    assert store.record_outcome(9999, "TP_HIT", 1.0, 1.0, 1.0) is False


def test_position_tracker_fills_outcome(store):
    tracker = PositionTracker(store=store)
    sid = insert_sample(store)
    tracker.open_position(
        "XAU/USD (Gold)", "BUY", 3000.0, 2980.0, 3040.0, "AI",
        signal_id=sid, cost_pct=0.05,
    )
    closed = tracker.check_sl_tp("XAU/USD (Gold)", 3041.0)
    assert closed is not None and closed.close_reason == "TP_HIT"

    row = store.fetch_all()[0]
    assert row["outcome"] == "TP_HIT"
    assert row["pnl_gross_pct"] == pytest.approx((3041 - 3000) / 3000 * 100)
    assert row["pnl_net_pct"] == pytest.approx(row["pnl_gross_pct"] - 0.05)


def test_report_aggregation(store):
    s1 = insert_sample(store)
    s2 = insert_sample(store, direction="SELL")
    s3 = insert_sample(store, instrument="BTC/USD (Bitcoin)")
    store.record_outcome(s1, "TP_HIT", 3040.0, 1.333, 1.0)
    store.record_outcome(s2, "SL_HIT", 3020.0, -0.667, -0.5)
    # s3 stays open (no outcome)

    report = store.compute_report()
    gold = report["XAU/USD (Gold)"]
    assert gold["signals"] == 2
    assert gold["closed"] == 2
    assert gold["win_rate"] == pytest.approx(0.5)
    assert gold["avg_net_expectancy_pct"] == pytest.approx(0.25)
    assert gold["hypothetical_return_pct"] == pytest.approx(
        ((1 + 0.01) * (1 - 0.005) - 1) * 100)
    assert gold["max_drawdown_pct"] == pytest.approx(-0.5)
    btc = report["BTC/USD (Bitcoin)"]
    assert btc["signals"] == 1
    assert btc["closed"] == 0


def test_csv_export(store, tmp_path):
    s1 = insert_sample(store)
    store.record_outcome(s1, "TP_HIT", 3040.0, 1.333, 1.0)
    out = tmp_path / "export.csv"
    n = store.export_csv(str(out))
    assert n == 1
    content = out.read_text()
    assert "XAU/USD (Gold)" in content
    assert "TP_HIT" in content
    assert content.splitlines()[0].startswith("id,ts_utc,instrument")


def test_model_version_missing_file():
    assert model_version("/nonexistent/model.pkl") == "nomodel"


def test_model_version_real_file(tmp_path):
    p = tmp_path / "m.pkl"
    p.write_bytes(b"fake model bytes")
    v1 = model_version(str(p))
    assert "@" in v1 and v1 != "nomodel"
    # Same content -> same hash prefix
    assert model_version(str(p)) == v1
    # Different content -> different version
    p.write_bytes(b"other model bytes")
    assert model_version(str(p)) != v1


# ---------------------------------------------------------------- v3.2 ----

def test_tier_is_persisted_and_exported(store, tmp_path):
    sid = store.insert_signal("XAU/USD (Gold)", "BUY", entry_price=1.0,
                              stop_loss=0.9, take_profit=1.2, tier="demo")
    row = store.fetch_all()[0]
    assert row["id"] == sid and row["tier"] == "demo"
    path = tmp_path / "x.csv"
    store.export_csv(str(path))
    assert "tier" in path.read_text().splitlines()[0]


def test_fetch_since_is_a_cursor(store):
    a = insert_sample(store)
    b = insert_sample(store, direction="SELL")
    c = insert_sample(store, instrument="BTC/USD (Bitcoin)")
    assert [r["id"] for r in store.fetch_since(0)] == [a, b, c]
    assert [r["id"] for r in store.fetch_since(a)] == [b, c]
    assert [r["id"] for r in store.fetch_since(a, instrument="BTC/USD (Bitcoin)")] == [c]
    assert store.fetch_since(c) == []


def test_old_database_is_migrated_with_tier_column(tmp_path):
    """A signals.db created before the tier column existed keeps working."""
    import sqlite3
    path = tmp_path / "old.db"
    conn = sqlite3.connect(path)
    conn.executescript("""
        CREATE TABLE signals (
            id INTEGER PRIMARY KEY AUTOINCREMENT, ts_utc TEXT NOT NULL,
            instrument TEXT NOT NULL, direction TEXT NOT NULL, confidence REAL,
            prob_buy REAL, prob_sell REAL, prob_hold REAL, adx REAL, regime TEXT,
            entry_price REAL, stop_loss REAL, take_profit REAL, strategy TEXT,
            model_version TEXT, outcome TEXT, outcome_ts_utc TEXT,
            exit_price REAL, pnl_gross_pct REAL, pnl_net_pct REAL);
        INSERT INTO signals (ts_utc, instrument, direction)
            VALUES ('2026-01-01T00:00:00Z', 'XAU/USD (Gold)', 'BUY');
    """)
    conn.commit(); conn.close()

    s = SignalStore(db_path=str(path))
    try:
        rows = s.fetch_all()
        assert rows[0]["tier"] is None            # legacy row, column added
        sid = s.insert_signal("XAU/USD (Gold)", "SELL", tier="demo")
        assert s.fetch_all()[-1]["tier"] == "demo" and sid == 2
    finally:
        s.close()
