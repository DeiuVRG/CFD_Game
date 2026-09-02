"""Sentinel tests - fully offline: fake broker, fake brain / fake Anthropic
client, temporary SQLite files. No gold_monitor or execution_capital import
(the sentinel must stay importable next to either)."""
import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from sentinel import rules
from sentinel.agent import Sentinel
from sentinel.brain import FALLBACK_BETA, ClaudeBrain, NullBrain, Research, Usage
from sentinel.config import InstrumentMap, SentinelConfig
from sentinel.rules import RiskState
from sentinel.schema import ManageDecision, OpenDecision, parse_manage, parse_open
from sentinel.signals_reader import SignalsReader
from sentinel.store import DecisionStore

GOLD = "XAU/USD (Gold)"
NOW = datetime(2026, 9, 2, 10, 0, tzinfo=timezone.utc)


def ts(dt):
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def signal_row(**over):
    row = dict(id=1, ts_utc=ts(NOW - timedelta(minutes=2)), instrument=GOLD,
               direction="BUY", confidence=0.66, prob_buy=0.66, prob_sell=0.1,
               prob_hold=0.24, adx=28.0, regime="TRENDING", entry_price=2400.0,
               stop_loss=2380.0, take_profit=2440.0, strategy="AI (66%)",
               model_version="abc@2026-09-01", tier="demo", outcome=None,
               pnl_net_pct=None)
    row.update(over)
    return row


def cfg(tmp_path, **over):
    c = SentinelConfig(signals_db=str(tmp_path / "signals.db"),
                       decisions_db=str(tmp_path / "decisions.db"),
                       instruments=[InstrumentMap(GOLD, "GOLD")],
                       discord_webhook="")
    for k, v in over.items():
        setattr(c, k, v)
    return c


# ---------------------------------------------------------------- stubs --

class FakeSignals:
    def __init__(self, rows=None):
        self.rows = list(rows or [])

    def fetch_since(self, last_id):
        return [r for r in self.rows if r["id"] > last_id]

    def latest_id(self):
        return max((r["id"] for r in self.rows), default=0)

    def stats(self, instrument):
        return {"signals": 3, "closed": 2, "win_rate": 0.5, "avg_net_pct": 0.1,
                "last_outcomes": ["BUY:TP_HIT:+1.00%"]}


class FakeBroker:
    def __init__(self, equity=10000.0, market_open=True):
        self._equity = equity
        self._positions = []
        self.market_open = market_open
        self.opened, self.closed, self.tightened = [], [], []
        self.next_status = "ACCEPTED"
        self._n = 0

    def equity(self):
        return self._equity

    def positions(self):
        return [dict(p) for p in self._positions]

    def market_info(self, epic):
        return {"epic": epic, "min_size": 0.01, "max_size": 100.0, "market_open": self.market_open}

    def candles(self, epic, resolution="HOUR", count=100):
        import pandas as pd
        rows, price = [], 2400.0
        for i in range(count):
            o = price; c = o + 1.0
            rows.append((o, c + 2, o - 2, c)); price = c
        df = pd.DataFrame(rows, columns=["open", "high", "low", "close"])
        df["timestamp"] = [NOW - timedelta(hours=count - i) for i in range(count)]
        return df

    def open(self, epic, direction, size, stop_loss, take_profit):
        self.opened.append((epic, direction, size, stop_loss, take_profit))
        if self.next_status != "ACCEPTED":
            return {"status": self.next_status, "deal_id": "", "level": 0, "reason": "nope"}
        self._n += 1
        deal = f"DEAL{self._n}"
        self._positions.append({"deal_id": deal, "epic": epic, "direction": direction,
                                "size": size, "open_level": 2401.0, "stop_level": stop_loss,
                                "profit_level": take_profit, "profit_loss": 0.0})
        return {"status": "ACCEPTED", "deal_id": deal, "level": 2401.0, "reason": ""}

    def close(self, deal_id):
        self.closed.append(deal_id)
        self._positions = [p for p in self._positions if p["deal_id"] != deal_id]
        return {"status": "CLOSED", "level": 2410.0, "profit": 9.0, "reason": ""}

    def tighten_sl(self, deal_id, new_sl):
        self.tightened.append((deal_id, new_sl))
        for p in self._positions:
            if p["deal_id"] == deal_id:
                p["stop_level"] = new_sl
        return {"status": "OK"}


class ScriptedBrain:
    def __init__(self, open_decision=None, manage_decision=None):
        self.open_decision = open_decision
        self.manage_decision = manage_decision
        self.research_calls = 0

    def research(self, instrument, snapshot):
        self.research_calls += 1
        return Research(brief="brief", usage=Usage(10, 5, "m"))

    def decide_open(self, context, research):
        return self.open_decision, Usage(100, 20, "claude-fable-5-1")

    def decide_manage(self, context, research):
        return self.manage_decision, Usage(80, 15, "claude-fable-5-1")


class Recorder:
    def __init__(self):
        self.sent = []

    def send(self, title, lines, color=0):
        self.sent.append((title, lines))
        return True


def approve(size=1.0, conf=0.8):
    return OpenDecision(action="APPROVE", size_fraction=size, confidence=conf,
                        rationale="ok", risks=["r1"])


def make_sentinel(tmp_path, rows, brain, broker=None, **cfg_over):
    c = cfg(tmp_path, **cfg_over)
    store = DecisionStore(c.decisions_db)
    broker = broker or FakeBroker()
    rec = Recorder()
    s = Sentinel(c, brain, broker, store, FakeSignals(rows), rec, now=lambda: NOW)
    return s, store, broker, rec


# ---------------------------------------------------------------- rules --

def base_state(**over):
    st = RiskState(equity=10000, day_start_equity=10000, trades_today=0, open_positions=[])
    for k, v in over.items():
        setattr(st, k, v)
    return st


def test_check_open_happy_path(tmp_path):
    ok, reason = rules.check_open(approve(), signal_row(), NOW, cfg(tmp_path), base_state(), "GOLD")
    assert (ok, reason) == (True, "OK")


@pytest.mark.parametrize("decision,state,row_over,expected", [
    (None, {}, {}, "NO_DECISION"),
    (OpenDecision(action="VETO", size_fraction=1, confidence=0.9, rationale="no"), {}, {}, "VETO"),
    (approve(size=0.0), {}, {}, "ZERO_SIZE"),
    (approve(), {}, {"ts_utc": ts(NOW - timedelta(hours=1))}, "STALE"),
    (approve(), {"equity": 9600}, {}, "LIMIT_DAILY_LOSS"),
    (approve(), {"trades_today": 5}, {}, "LIMIT_TRADES"),
    (approve(), {"open_positions": [{"epic": "A"}, {"epic": "B"}]}, {}, "LIMIT_POSITIONS"),
    (approve(), {"open_positions": [{"epic": "GOLD"}]}, {}, "DUPLICATE_EPIC"),
    (approve(), {}, {"stop_loss": 2400.0}, "INVALID_SL"),
    (approve(), {}, {"take_profit": 2405.0}, "LOW_RR"),
])
def test_check_open_rejections(tmp_path, decision, state, row_over, expected):
    ok, reason = rules.check_open(decision, signal_row(**row_over), NOW, cfg(tmp_path),
                                  base_state(**state), "GOLD")
    assert ok is False and reason == expected


def test_check_manage_only_tightens_in_the_positions_favour():
    buy = {"direction": "BUY", "stop_level": 2380.0}
    sell = {"direction": "SELL", "stop_level": 2420.0}
    t = lambda sl: ManageDecision(action="TIGHTEN_SL", new_stop_loss=sl, confidence=0.7, rationale="x")
    assert rules.check_manage(t(2390.0), buy, 2410.0) == ("TIGHTEN_SL", 2390.0, "OK")
    assert rules.check_manage(t(2370.0), buy, 2410.0)[0] == "HOLD"     # widening
    assert rules.check_manage(t(2415.0), buy, 2410.0)[0] == "HOLD"     # beyond price
    assert rules.check_manage(t(2410.0), sell, 2400.0) == ("TIGHTEN_SL", 2410.0, "OK")
    assert rules.check_manage(t(2430.0), sell, 2400.0)[0] == "HOLD"
    assert rules.check_manage(t(None), buy, 2410.0) == ("HOLD", None, "INVALID_SL")
    assert rules.check_manage(None, buy, 2410.0) == ("HOLD", None, "NO_DECISION")
    close = ManageDecision(action="CLOSE", confidence=0.9, rationale="x")
    assert rules.check_manage(close, buy, 2410.0) == ("CLOSE", None, "OK")


def test_position_size():
    assert rules.position_size(10000, 0.01, 2400, 2380, 0.01, 100) == 5.0      # 100 / 20
    assert rules.position_size(10000, 0.01, 2400, 2380, 0.01, 2.0) == 2.0       # clamped
    assert rules.position_size(10000, 0.01, 2400, 2380, 10.0, 100) == 0.0       # below min
    assert rules.position_size(10000, 0.0, 2400, 2380, 0.01, 100) == 0.0


# --------------------------------------------------------------- schema --

def test_schema_parsing():
    d = parse_open(json.dumps({"action": "APPROVE", "size_fraction": 0.5, "confidence": 0.7,
                               "rationale": "ok", "risks": []}))
    assert d.size_fraction == 0.5
    assert parse_open('{"action": "MAYBE"}') is None
    assert parse_open("not json") is None
    m = parse_manage(json.dumps({"action": "TIGHTEN_SL", "new_stop_loss": 2390, "confidence": 0.6,
                                 "rationale": "x"}))
    assert m.new_stop_loss == 2390
    assert parse_manage(json.dumps({"action": "CLOSE", "new_stop_loss": None, "confidence": 2,
                                    "rationale": "x"})) is None


# ---------------------------------------------------------------- store --

def test_store_state_cursor_outcomes_and_trades_today(tmp_path):
    store = DecisionStore(str(tmp_path / "d.db"))
    assert store.get_state("last_signal_id") is None
    store.set_state("last_signal_id", 7); store.set_state("last_signal_id", 9)
    assert store.get_state("last_signal_id") == "9"
    a = store.insert_decision(kind="OPEN", final_action="OPEN", deal_id="D1", epic="GOLD",
                              llm_risks=["x", "y"])
    store.insert_decision(kind="OPEN", final_action="VETO", epic="GOLD")
    assert store.trades_today() == 1
    assert [t["id"] for t in store.open_trades()] == [a]
    assert json.loads(store.fetch_all()[0]["llm_risks"]) == ["x", "y"]
    assert store.record_outcome(a, "CLOSED_AT_BROKER", 12.5) is True
    assert store.record_outcome(a, "AGAIN", 0) is False           # once only
    assert store.open_trades() == []
    rid = store.insert_research(GOLD, "brief", "m", 1, 2)
    assert store.latest_research(GOLD, 3600)["id"] == rid
    assert store.latest_research(GOLD, 0, now=datetime.now(timezone.utc) + timedelta(seconds=5)) is None


# --------------------------------------------------------- signals reader --

def test_signals_reader_reads_gold_monitor_db(tmp_path):
    import sqlite3
    path = tmp_path / "signals.db"
    conn = sqlite3.connect(path)
    conn.executescript("""
        CREATE TABLE signals (id INTEGER PRIMARY KEY AUTOINCREMENT, ts_utc TEXT, instrument TEXT,
            direction TEXT, outcome TEXT, pnl_net_pct REAL);
        INSERT INTO signals (ts_utc, instrument, direction, outcome, pnl_net_pct) VALUES
            ('2026-09-01T09:00:00Z', 'XAU/USD (Gold)', 'BUY', 'TP_HIT', 1.2),
            ('2026-09-01T12:00:00Z', 'XAU/USD (Gold)', 'SELL', 'SL_HIT', -0.8),
            ('2026-09-01T15:00:00Z', 'XAU/USD (Gold)', 'BUY', NULL, NULL);
    """)
    conn.commit(); conn.close()
    r = SignalsReader(str(path))
    assert r.latest_id() == 3
    assert [x["id"] for x in r.fetch_since(1)] == [2, 3]
    st = r.stats(GOLD)
    assert st["signals"] == 3 and st["closed"] == 2 and st["win_rate"] == 0.5
    assert st["avg_net_pct"] == pytest.approx(0.2)
    missing = SignalsReader(str(tmp_path / "nope.db"))
    assert missing.fetch_since(0) == [] and missing.stats(GOLD)["signals"] == 0


# ---------------------------------------------------------------- agent --

def test_approved_signal_opens_demo_position(tmp_path):
    s, store, broker, rec = make_sentinel(tmp_path, [signal_row()], ScriptedBrain(approve(size=0.5)))
    s.run_once()
    assert broker.opened == [("GOLD", "BUY", 2.5, 2380.0, 2440.0)]   # 10000*1%*0.5 / 20
    rows = store.fetch_all(kind="OPEN")     # run_once also runs the first review (HOLD)
    assert len(rows) == 1 and rows[0]["final_action"] == "OPEN" and rows[0]["deal_id"] == "DEAL1"
    assert rows[0]["llm_action"] == "APPROVE" and rows[0]["input_tokens"] == 100
    assert rows[0]["research_id"] is not None and rows[0]["llm_rationale"] == "ok"
    assert store.get_state("last_signal_id") == "1"
    assert store.get_state(f"day_start_equity:{NOW:%Y-%m-%d}") == "10000.0"
    assert rec.sent[0][0].startswith("DEMO: pozitie deschisa")


def test_veto_and_rejections_place_no_order(tmp_path):
    veto = OpenDecision(action="VETO", size_fraction=1, confidence=0.9, rationale="event risk")
    s, store, broker, rec = make_sentinel(tmp_path, [signal_row()], ScriptedBrain(veto))
    s.run_once()
    assert broker.opened == []
    assert store.fetch_all()[0]["final_action"] == "VETO"
    assert store.fetch_all()[0]["llm_rationale"] == "event risk"

    # model unavailable -> fail closed
    s2, store2, broker2, _ = make_sentinel(tmp_path / "b", [signal_row()], ScriptedBrain(None))
    s2.run_once()
    assert broker2.opened == [] and store2.fetch_all()[0]["reason"] == "NO_DECISION"


def test_stale_signal_is_skipped_without_calling_the_model(tmp_path):
    brain = ScriptedBrain(approve())
    old = signal_row(ts_utc=ts(NOW - timedelta(hours=2)))
    s, store, broker, _ = make_sentinel(tmp_path, [old], brain)
    s.run_once()
    assert brain.research_calls == 0 and broker.opened == []
    assert store.fetch_all()[0]["final_action"] == "SKIP"
    assert store.get_state("last_signal_id") == "1"


def test_dry_run_logs_but_never_orders(tmp_path):
    s, store, broker, rec = make_sentinel(tmp_path, [signal_row()], ScriptedBrain(approve()),
                                          dry_run=True)
    s.run_once()
    assert broker.opened == []
    row = store.fetch_all()[0]
    assert row["final_action"] == "DRY_RUN" and row["dry_run"] == 1 and row["size"] == 5.0
    assert store.open_trades() == []


def test_market_closed_and_broker_rejection(tmp_path):
    s, store, broker, _ = make_sentinel(tmp_path, [signal_row()], ScriptedBrain(approve()),
                                        broker=FakeBroker(market_open=False))
    s.run_once()
    assert broker.opened == [] and store.fetch_all()[0]["reason"] == "MARKET_CLOSED"

    b = FakeBroker(); b.next_status = "REJECTED"
    s2, store2, _, _ = make_sentinel(tmp_path / "b", [signal_row()], ScriptedBrain(approve()), broker=b)
    s2.run_once()
    assert store2.fetch_all()[0]["final_action"] == "REJECT"
    assert store2.fetch_all()[0]["reason"].startswith("BROKER:REJECTED")


def test_hard_limits_beat_the_model(tmp_path):
    rows = [signal_row(id=i, ts_utc=ts(NOW - timedelta(minutes=i)),
                       instrument=GOLD) for i in range(1, 4)]
    s, store, broker, _ = make_sentinel(tmp_path, rows, ScriptedBrain(approve()),
                                        max_concurrent_positions=1)
    s.run_once()
    finals = [r["final_action"] for r in store.fetch_all(kind="OPEN")]
    reasons = [r["reason"] for r in store.fetch_all(kind="OPEN")]
    assert finals == ["OPEN", "REJECT", "REJECT"]
    assert reasons[1] == "LIMIT_POSITIONS" and len(broker.opened) == 1


def test_research_is_cached_within_ttl(tmp_path):
    brain = ScriptedBrain(OpenDecision(action="VETO", size_fraction=1, confidence=0.5, rationale="n"))
    rows = [signal_row(id=1), signal_row(id=2, ts_utc=ts(NOW - timedelta(minutes=1)))]
    s, store, _, _ = make_sentinel(tmp_path, rows, brain)
    s.run_once()
    assert brain.research_calls == 1
    assert {r["research_id"] for r in store.fetch_all()} == {1}


def test_review_tighten_close_and_hold(tmp_path):
    tighten = ManageDecision(action="TIGHTEN_SL", new_stop_loss=2395.0, confidence=0.7, rationale="lock")
    s, store, broker, rec = make_sentinel(tmp_path, [signal_row()],
                                          ScriptedBrain(approve(), tighten))
    s.run_once()                              # opens DEAL1, first review runs immediately
    assert broker.tightened == [("DEAL1", 2395.0)]
    review = [r for r in store.fetch_all() if r["kind"] == "REVIEW"]
    assert review[-1]["final_action"] == "TIGHTEN_SL" and review[-1]["stop_loss"] == 2395.0

    # widening is rejected -> HOLD
    s.brain.manage_decision = ManageDecision(action="TIGHTEN_SL", new_stop_loss=2300.0,
                                             confidence=0.7, rationale="bad")
    s.review_positions(force=True)
    assert broker.tightened == [("DEAL1", 2395.0)]
    assert store.fetch_all()[-1]["final_action"] == "HOLD" and store.fetch_all()[-1]["reason"] == "INVALID_SL"

    # close
    s.brain.manage_decision = ManageDecision(action="CLOSE", confidence=0.9, rationale="news")
    s.review_positions(force=True)
    assert broker.closed == ["DEAL1"]
    assert store.fetch_all()[-1]["final_action"] == "CLOSE"
    opened = store.fetch_all(kind="OPEN")[0]
    assert opened["outcome"] == "CLOSED_BY_SENTINEL" and opened["pnl"] == 9.0
    assert store.open_trades() == []


def test_review_respects_interval(tmp_path):
    hold = ManageDecision(action="HOLD", confidence=0.5, rationale="h")
    s, store, broker, _ = make_sentinel(tmp_path, [signal_row()], ScriptedBrain(approve(), hold))
    s.run_once()
    n = len(store.fetch_all(kind="REVIEW"))
    s.review_positions()                      # same NOW -> interval not elapsed
    assert len(store.fetch_all(kind="REVIEW")) == n
    s.review_positions(force=True)
    assert len(store.fetch_all(kind="REVIEW")) == n + 1


def test_position_closed_at_broker_gets_an_outcome(tmp_path):
    hold = ManageDecision(action="HOLD", confidence=0.5, rationale="h")
    s, store, broker, rec = make_sentinel(tmp_path, [signal_row()], ScriptedBrain(approve(), hold))
    s.run_once()
    broker._positions[0]["profit_loss"] = 42.0
    s.run_once()                              # records last known pnl
    broker._positions.clear()                 # TP/SL hit at the broker
    s.run_once()
    opened = store.fetch_all(kind="OPEN")[0]
    assert opened["outcome"] == "CLOSED_AT_BROKER" and opened["pnl"] == 42.0
    assert any(t.startswith("Pozitie inchisa la broker") for t, _ in rec.sent)


def test_unknown_instrument_is_ignored_but_cursor_advances(tmp_path):
    s, store, broker, _ = make_sentinel(tmp_path, [signal_row(instrument="EUR/USD")],
                                        ScriptedBrain(approve()))
    s.run_once()
    assert store.fetch_all() == [] and store.get_state("last_signal_id") == "1"


def test_null_brain_is_a_deterministic_control_group(tmp_path):
    s, store, broker, _ = make_sentinel(tmp_path, [signal_row()], NullBrain())
    s.run_once()
    assert broker.opened[0][2] == 5.0
    assert store.fetch_all()[0]["llm_rationale"].startswith("NullBrain")


# ---------------------------------------------------------------- brain --

class FakeMessages:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self.responses.pop(0)


def fake_client(*responses):
    msgs = FakeMessages(responses)
    return SimpleNamespace(beta=SimpleNamespace(messages=msgs)), msgs


def resp(text=None, stop_reason="end_turn", content=None, category=None):
    blocks = content if content is not None else [SimpleNamespace(type="text", text=text or "")]
    return SimpleNamespace(
        stop_reason=stop_reason, content=blocks, model="claude-fable-5-1",
        usage=SimpleNamespace(input_tokens=321, output_tokens=45),
        stop_details=SimpleNamespace(category=category) if category else None)


def test_claude_brain_decide_open_uses_structured_output_and_fallbacks(tmp_path):
    body = {"action": "APPROVE", "size_fraction": 0.7, "confidence": 0.8,
            "rationale": "trend + fara evenimente", "risks": ["CPI maine"]}
    client, msgs = fake_client(resp(json.dumps(body)))
    brain = ClaudeBrain(cfg(tmp_path), client=client)
    decision, usage = brain.decide_open("{ctx}", "brief")
    assert decision.action == "APPROVE" and decision.size_fraction == 0.7
    assert usage.input_tokens == 321 and usage.model == "claude-fable-5-1"
    kw = msgs.calls[0]
    assert kw["model"] == "claude-fable-5-1"
    assert kw["betas"] == [FALLBACK_BETA] and kw["fallbacks"] == "default"
    assert kw["output_config"]["format"]["type"] == "json_schema"
    assert kw["output_config"]["effort"] == "high"
    assert "thinking" not in kw                      # always on for Fable 5.1
    assert kw["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert "brief" in kw["messages"][0]["content"]


def test_claude_brain_refusal_and_garbage_yield_none(tmp_path):
    client, _ = fake_client(resp(stop_reason="refusal", content=[], category="cyber"),
                            resp("not json at all"))
    brain = ClaudeBrain(cfg(tmp_path), client=client)
    assert brain.decide_open("c", "r")[0] is None
    assert brain.decide_manage("c", "r")[0] is None


def test_claude_brain_research_uses_web_search_and_resumes_pause_turn(tmp_path):
    paused = resp(stop_reason="pause_turn",
                  content=[SimpleNamespace(type="server_tool_use", name="web_search")])
    final = resp("Gold: CPI tomorrow 12:30 UTC...")
    client, msgs = fake_client(paused, final)
    brain = ClaudeBrain(cfg(tmp_path), client=client)
    r = brain.research(GOLD, "{snapshot}")
    assert r.brief.startswith("Gold: CPI") and r.refused is False
    assert len(msgs.calls) == 2
    tools = msgs.calls[0]["tools"]
    assert tools[0]["type"] == "web_search_20260209" and tools[0]["max_uses"] == 4
    assert msgs.calls[1]["messages"][1]["role"] == "assistant"     # paused turn re-sent
    assert msgs.calls[0]["output_config"] == {"effort": "medium"}


def test_claude_brain_research_without_web_search(tmp_path):
    client, msgs = fake_client(resp("brief"))
    brain = ClaudeBrain(cfg(tmp_path, web_search=False), client=client)
    brain.research(GOLD, "s")
    assert msgs.calls[0]["tools"] == []


# ------------------------------------------------------- Agent SDK brain --

class FakeResult:
    """Duck-typed claude_agent_sdk.ResultMessage."""
    def __init__(self, result=None, structured_output=None, is_error=False,
                 stop_reason="end_turn", usage=None):
        self.result, self.structured_output, self.is_error = result, structured_output, is_error
        self.stop_reason = stop_reason
        self.usage = usage if usage is not None else {"input_tokens": 12, "output_tokens": 34}


def fake_query(results):
    """query_fn that records (prompt, options) and yields scripted results."""
    calls = []
    queue = list(results)

    async def query(*, prompt, options=None, **_):
        calls.append((prompt, options))
        yield SimpleNamespace(type="assistant")
        yield queue.pop(0)
    query.calls = calls
    return query


def sdk_brain(tmp_path, results, **cfg_over):
    from sentinel.brain_sdk import AgentSdkBrain
    q = fake_query(results)
    return AgentSdkBrain(cfg(tmp_path, **cfg_over), query_fn=q, cwd=str(tmp_path / "cwd")), q


def test_agent_sdk_brain_decision_uses_structured_output(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-should-be-removed")
    body = {"action": "VETO", "size_fraction": 0.0, "confidence": 0.9,
            "rationale": "CPI in 20 min", "risks": ["event"]}
    brain, q = sdk_brain(tmp_path, [FakeResult(result=json.dumps(body), structured_output=body)])
    assert "ANTHROPIC_API_KEY" not in __import__("os").environ      # subscription path
    decision, usage = brain.decide_open("{ctx}", "brief")
    assert decision.action == "VETO" and decision.rationale == "CPI in 20 min"
    assert usage.input_tokens == 12 and usage.model == "claude-fable-5-1"
    prompt, opts = q.calls[0]
    assert "brief" in prompt and "{ctx}" in prompt
    assert opts.model == "claude-fable-5-1" and opts.tools == [] and opts.max_turns == 3
    assert opts.output_format["schema"]["required"] == ["action", "size_fraction", "confidence", "rationale", "risks"]
    assert opts.setting_sources == [] and opts.effort == "high"
    assert opts.system_prompt.startswith("You are the sentinel")


def test_agent_sdk_brain_research_uses_web_tools(tmp_path):
    brain, q = sdk_brain(tmp_path, [FakeResult(result="  Gold: FOMC tomorrow.  ")])
    r = brain.research(GOLD, "{snap}")
    assert r.brief == "Gold: FOMC tomorrow." and r.refused is False
    _, opts = q.calls[0]
    assert opts.tools == ["WebSearch", "WebFetch"] and opts.allowed_tools == ["WebSearch", "WebFetch"]
    assert opts.max_turns == 8 and opts.effort == "medium" and opts.output_format is None


def test_agent_sdk_brain_fails_closed(tmp_path):
    brain, _ = sdk_brain(tmp_path, [FakeResult(is_error=True, result="boom"),
                                    FakeResult(stop_reason="refusal"),
                                    FakeResult(result="garbage", structured_output=None)])
    assert brain.decide_open("c", "r")[0] is None
    assert brain.decide_manage("c", "r")[0] is None
    assert brain.decide_open("c", "r")[0] is None

    async def broken(*, prompt, options=None, **_):
        raise RuntimeError("CLI not found")
        yield  # pragma: no cover
    from sentinel.brain_sdk import AgentSdkBrain
    b = AgentSdkBrain(cfg(tmp_path), query_fn=broken, cwd=str(tmp_path / "cwd2"))
    assert b.research(GOLD, "s").brief == "" and b.decide_open("c", "r")[0] is None
