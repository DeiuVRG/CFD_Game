"""The sentinel loop.

    new signal in signals.db -> research (cached) -> model APPROVE/VETO
      -> hard rules -> size -> demo order -> decisions.db -> Discord
    every review interval: open position -> model HOLD/CLOSE/TIGHTEN_SL
      -> hard rules -> demo order -> decisions.db -> Discord
    every poll: positions that disappeared from the broker -> outcome

All model outputs pass through rules.check_open / rules.check_manage; on
any model failure the agent fails closed (no trade, no change).
"""
import logging
import time
from datetime import datetime, timezone
from typing import Optional

from sentinel import rules
from sentinel.config import InstrumentMap, SentinelConfig
from sentinel.context import indicator_snapshot, render_manage_context, render_open_context
from sentinel.notify import COLOR_CLOSE, COLOR_INFO, COLOR_OPEN, COLOR_VETO, NullNotifier
from sentinel.rules import RiskState
from sentinel.store import DecisionStore

logger = logging.getLogger(__name__)

ACCEPTED = ("ACCEPTED", "OPEN")


class Sentinel:
    def __init__(self, cfg: SentinelConfig, brain, broker, store: DecisionStore,
                 signals, notifier=None, now=None):
        self.cfg = cfg
        self.brain = brain
        self.broker = broker
        self.store = store
        self.signals = signals
        self.notifier = notifier or NullNotifier()
        self._now = now or (lambda: datetime.now(timezone.utc))
        self._last_review: Optional[datetime] = None
        self._equity: float = 0.0
        self._positions: list = []

    # ------------------------------------------------------------ helpers
    def now(self) -> datetime:
        return self._now()

    def now_iso(self) -> str:
        return self.now().strftime("%Y-%m-%dT%H:%M:%SZ")

    def _record(self, **fields) -> int:
        fields.setdefault("ts_utc", self.now_iso())
        return self.store.insert_decision(**fields)

    def _limits(self) -> dict:
        c = self.cfg
        return {"max_risk_per_trade": c.max_risk_per_trade, "max_daily_loss": c.max_daily_loss,
                "max_trades_per_day": c.max_trades_per_day,
                "max_concurrent_positions": c.max_concurrent_positions}

    def _account(self) -> dict:
        st = self.risk_state()
        return {"equity": round(st.equity, 2), "day_start_equity": round(st.day_start_equity, 2),
                "daily_loss_pct": round(st.daily_loss_pct * 100, 2),
                "trades_today": st.trades_today,
                "open_positions": [{"epic": p["epic"], "direction": p["direction"],
                                    "size": p["size"], "pnl": p.get("profit_loss")}
                                   for p in st.open_positions],
                "mode": "DRY_RUN" if self.cfg.dry_run else "DEMO"}

    def risk_state(self) -> RiskState:
        day = self.now().strftime("%Y-%m-%d")
        start = self.store.get_state(f"day_start_equity:{day}")
        return RiskState(
            equity=self._equity,
            day_start_equity=float(start) if start is not None else self._equity,
            trades_today=self.store.trades_today(day),
            open_positions=list(self._positions),
        )

    # ------------------------------------------------------------ account
    def refresh_account(self):
        self._equity = float(self.broker.equity())
        self._positions = list(self.broker.positions())
        key = f"day_start_equity:{self.now().strftime('%Y-%m-%d')}"
        if self.store.get_state(key) is None:
            self.store.set_state(key, self._equity)

    def sync_closed(self):
        """Executed trades that are no longer open at the broker -> outcome."""
        open_ids = {p["deal_id"]: p for p in self._positions}
        for trade in self.store.open_trades():
            pos = open_ids.get(trade["deal_id"])
            if pos is not None:
                self.store.update_last_pnl(trade["id"], pos.get("profit_loss"))
                continue
            pnl = trade.get("last_pnl")
            self.store.record_outcome(trade["id"], "CLOSED_AT_BROKER", pnl, ts_utc=self.now_iso())
            self.notifier.send(
                f"Pozitie inchisa la broker: {trade['direction']} {trade['epic']}",
                [f"Ultimul P&L cunoscut: {pnl}", f"Decizie #{trade['id']}, semnal #{trade['signal_id']}"],
                COLOR_CLOSE)

    # ------------------------------------------------------------ signals
    def process_signals(self):
        cursor = int(self.store.get_state("last_signal_id", 0))
        for row in self.signals.fetch_since(cursor):
            try:
                self._handle_signal(row)
            except Exception as e:   # one bad signal must not stall the cursor
                logger.error(f"signal {row.get('id')} failed: {e}", exc_info=True)
            finally:
                self.store.set_state("last_signal_id", row["id"])

    def _research(self, imap: InstrumentMap, snapshot: dict) -> tuple:
        cached = self.store.latest_research(imap.signal_name, self.cfg.research_ttl_sec,
                                            now=self.now())
        if cached is not None:
            return cached["brief"] or "", cached["id"]
        res = self.brain.research(imap.signal_name, str(snapshot))
        rid = self.store.insert_research(imap.signal_name, res.brief, res.usage.model,
                                         res.usage.input_tokens, res.usage.output_tokens,
                                         ts_utc=self.now_iso())
        return res.brief, rid

    def _snapshot(self, imap: InstrumentMap) -> dict:
        try:
            df = self.broker.candles(imap.epic, imap.resolution, self.cfg.candles_for_context + 30)
            return indicator_snapshot(df)
        except Exception as e:
            logger.warning(f"candles for {imap.epic} unavailable: {e}")
            return {}

    def _handle_signal(self, row: dict):
        imap = self.cfg.epic_for(row["instrument"])
        if imap is None:
            return
        now = self.now()
        base = dict(kind="OPEN", instrument=row["instrument"], epic=imap.epic,
                    signal_id=row["id"], direction=row["direction"],
                    entry_price=row["entry_price"], stop_loss=row["stop_loss"],
                    take_profit=row["take_profit"], dry_run=self.cfg.dry_run)

        if rules.signal_age_sec(row, now) > self.cfg.signal_max_age_sec:
            self._record(final_action="SKIP", reason="STALE", **base)
            logger.info(f"signal {row['id']} stale - skipped")
            return

        snapshot = self._snapshot(imap)
        brief, research_id = self._research(imap, snapshot)
        context = render_open_context(row, imap.epic, snapshot,
                                      self.signals.stats(row["instrument"]),
                                      self._account(), self._limits())
        decision, usage = self.brain.decide_open(context, brief)
        llm = {}
        if decision is not None:
            llm = dict(llm_action=decision.action, llm_size_fraction=decision.size_fraction,
                       llm_confidence=decision.confidence, llm_rationale=decision.rationale,
                       llm_risks=decision.risks)
        meta = dict(model=usage.model or self.cfg.model, input_tokens=usage.input_tokens,
                    output_tokens=usage.output_tokens, research_id=research_id)

        ok, reason = rules.check_open(decision, row, now, self.cfg, self.risk_state(), imap.epic)
        if not ok:
            final = "VETO" if reason == "VETO" else "REJECT"
            self._record(final_action=final, reason=reason, **base, **llm, **meta)
            self._notify_decision(row, imap, final, reason, decision)
            return

        info = self.broker.market_info(imap.epic) or {}
        if not info.get("market_open", False):
            self._record(final_action="REJECT", reason="MARKET_CLOSED",
                                       **base, **llm, **meta)
            return
        size = rules.position_size(
            self._equity, self.cfg.max_risk_per_trade * decision.size_fraction,
            row["entry_price"], row["stop_loss"],
            info.get("min_size", 0.0), info.get("max_size", float("inf")))
        if size <= 0:
            self._record(final_action="REJECT", reason="SIZE_ZERO",
                                       **base, **llm, **meta)
            return

        if self.cfg.dry_run:
            self._record(final_action="DRY_RUN", reason="OK", size=size,
                                       **base, **llm, **meta)
            self._notify_decision(row, imap, "DRY_RUN", f"size {size}", decision)
            return

        conf = self.broker.open(imap.epic, row["direction"], size,
                                row["stop_loss"], row["take_profit"])
        if conf.get("status") not in ACCEPTED or not conf.get("deal_id"):
            self._record(final_action="REJECT", reason=f"BROKER:{conf.get('status')}:{conf.get('reason')}",
                                       size=size, **base, **llm, **meta)
            self._notify_decision(row, imap, "REJECT", f"broker {conf.get('status')} {conf.get('reason')}", decision)
            return
        self._record(final_action="OPEN", reason="OK", size=size,
                                   deal_id=conf["deal_id"], **base, **llm, **meta)
        self._positions.append({"deal_id": conf["deal_id"], "epic": imap.epic,
                                "direction": row["direction"], "size": size,
                                "open_level": conf.get("level") or row["entry_price"],
                                "stop_level": row["stop_loss"], "profit_level": row["take_profit"],
                                "profit_loss": 0.0})
        self._notify_decision(row, imap, "OPEN", f"size {size}, deal {conf['deal_id']}", decision)

    def _notify_decision(self, row, imap, final, detail, decision):
        title = {"OPEN": "DEMO: pozitie deschisa", "DRY_RUN": "DRY-RUN: ar fi deschis",
                 "VETO": "Semnal respins de santinela", "REJECT": "Semnal blocat de reguli"}.get(final, final)
        color = COLOR_OPEN if final in ("OPEN", "DRY_RUN") else COLOR_VETO
        lines = [f"{row['direction']} {imap.epic} @ {row['entry_price']} | SL {row['stop_loss']} | TP {row['take_profit']}",
                 f"Rezultat: {final} ({detail})"]
        if decision is not None:
            lines.append(f"Model: {decision.action}, size x{getattr(decision, 'size_fraction', 1):.2f}, "
                         f"incredere {decision.confidence:.0%}")
            lines.append(f"Motiv: {decision.rationale}")
        self.notifier.send(title, lines, color)

    # ------------------------------------------------------------ review
    def review_positions(self, force: bool = False):
        now = self.now()
        if not force and self._last_review is not None and \
                (now - self._last_review).total_seconds() < self.cfg.review_interval_sec:
            return
        self._last_review = now
        by_deal = {p["deal_id"]: p for p in self._positions}
        for trade in self.store.open_trades():
            pos = by_deal.get(trade["deal_id"])
            if pos is None:
                continue
            imap = self.cfg.epic_for(trade["instrument"])
            if imap is None:
                continue
            snapshot = self._snapshot(imap)
            price = snapshot.get("last_close") or pos.get("open_level") or 0.0
            cached = self.store.latest_research(imap.signal_name, self.cfg.research_ttl_sec, now=now)
            brief = (cached or {}).get("brief") or ""
            context = render_manage_context(trade, pos, snapshot, self._account(),
                                            now.strftime("%Y-%m-%dT%H:%M:%SZ"))
            decision, usage = self.brain.decide_manage(context, brief)
            action, new_sl, reason = rules.check_manage(decision, pos, price)
            base = dict(kind="REVIEW", instrument=trade["instrument"], epic=imap.epic,
                        signal_id=trade["signal_id"], deal_id=trade["deal_id"],
                        direction=pos["direction"], entry_price=pos.get("open_level"),
                        stop_loss=new_sl if action == "TIGHTEN_SL" else pos.get("stop_level"),
                        take_profit=pos.get("profit_level"), size=pos.get("size"),
                        model=usage.model or self.cfg.model, input_tokens=usage.input_tokens,
                        output_tokens=usage.output_tokens, dry_run=self.cfg.dry_run,
                        research_id=(cached or {}).get("id"))
            llm = {}
            if decision is not None:
                llm = dict(llm_action=decision.action, llm_confidence=decision.confidence,
                           llm_rationale=decision.rationale)

            if action == "CLOSE":
                res = self.broker.close(trade["deal_id"])
                ok = res.get("status") not in ("ERROR",)
                self._record(final_action="CLOSE" if ok else "HOLD",
                                           reason="OK" if ok else f"BROKER:{res.get('reason')}",
                                           **base, **llm)
                if ok:
                    self.store.record_outcome(trade["id"], "CLOSED_BY_SENTINEL",
                                              res.get("profit", pos.get("profit_loss")),
                                              ts_utc=self.now_iso())
                    self.notifier.send(f"DEMO: inchis de santinela {pos['direction']} {imap.epic}",
                                       [f"P&L: {res.get('profit', pos.get('profit_loss'))}",
                                        f"Motiv: {decision.rationale if decision else reason}"],
                                       COLOR_CLOSE)
            elif action == "TIGHTEN_SL":
                res = self.broker.tighten_sl(trade["deal_id"], new_sl)
                ok = res.get("status") != "ERROR"
                self._record(final_action="TIGHTEN_SL" if ok else "HOLD",
                                           reason="OK" if ok else f"BROKER:{res.get('reason')}",
                                           **base, **llm)
                if ok:
                    pos["stop_level"] = new_sl
                    self.notifier.send(f"DEMO: SL strans {pos['direction']} {imap.epic}",
                                       [f"SL nou: {new_sl}", f"Motiv: {decision.rationale if decision else reason}"],
                                       COLOR_INFO)
            else:
                self._record(final_action="HOLD", reason=reason, **base, **llm)

    # ------------------------------------------------------------ loop
    def run_once(self):
        self.refresh_account()
        self.sync_closed()
        self.process_signals()
        self.review_positions()

    def run_forever(self):
        logger.info(f"Sentinel started (mode={'DRY_RUN' if self.cfg.dry_run else 'DEMO'}, "
                    f"model={self.cfg.model})")
        while True:
            try:
                self.run_once()
            except KeyboardInterrupt:
                logger.info("Sentinel stopped.")
                return
            except Exception as e:
                logger.error(f"cycle failed: {e}", exc_info=True)
            time.sleep(self.cfg.poll_interval_sec)
