#!/usr/bin/env python3
"""Sentinel CLI.

    python -m sentinel.main --run              # loop (DEMO account)
    python -m sentinel.main --once             # one cycle
    python -m sentinel.main --run --dry-run    # decisions logged, no orders
    python -m sentinel.main --run --no-llm     # deterministic control group
    python -m sentinel.main --markets gold     # find the Capital.com epic
    python -m sentinel.main --report           # decision log summary
"""
import argparse
import logging
import os
import sys

from sentinel.config import SENTINEL_DIR, SentinelConfig


def setup_logging():
    os.makedirs(os.path.join(SENTINEL_DIR, "logs"), exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.FileHandler(os.path.join(SENTINEL_DIR, "logs", "sentinel.log"),
                                      encoding="utf-8"),
                  logging.StreamHandler(sys.stdout)],
    )


def build(cfg: SentinelConfig, no_llm: bool):
    from sentinel.agent import Sentinel
    from sentinel.brain import ClaudeBrain, NullBrain
    from sentinel.broker import DemoBroker
    from sentinel.notify import DiscordNotifier
    from sentinel.signals_reader import SignalsReader
    from sentinel.store import DecisionStore

    if no_llm:
        brain = NullBrain()
    elif cfg.brain == "api":
        brain = ClaudeBrain(cfg)
    else:
        from sentinel.brain_sdk import AgentSdkBrain
        brain = AgentSdkBrain(cfg)
    logging.getLogger(__name__).info(f"brain: {type(brain).__name__} (model {cfg.model})")
    broker = DemoBroker(cfg)
    store = DecisionStore(cfg.decisions_db)
    if store.get_state("last_signal_id") is None:
        # First start: never replay history - only signals emitted from now on.
        store.set_state("last_signal_id", SignalsReader(cfg.signals_db).latest_id())
    return Sentinel(cfg, brain, broker, store, SignalsReader(cfg.signals_db),
                    DiscordNotifier(cfg.discord_webhook))


def cmd_report(cfg: SentinelConfig):
    from sentinel.store import DecisionStore
    store = DecisionStore(cfg.decisions_db)
    rows = store.fetch_all()
    if not rows:
        print("No decisions yet.")
        return
    by_action = {}
    for r in rows:
        by_action[r["final_action"]] = by_action.get(r["final_action"], 0) + 1
    print(f"\n  SENTINEL DECISIONS: {len(rows)} rows")
    for k, v in sorted(by_action.items()):
        print(f"    {k:<12} {v}")
    closed = [r for r in rows if r["kind"] == "OPEN" and r["outcome"] is not None]
    if closed:
        pnl = [r["pnl"] or 0.0 for r in closed]
        wins = sum(1 for p in pnl if p > 0)
        print(f"\n  Closed demo trades: {len(closed)}  wins: {wins}  "
              f"win rate: {wins / len(closed):.0%}  total P&L: {sum(pnl):+.2f}")
    vetoed = [r for r in rows if r["final_action"] == "VETO"]
    print(f"  Vetoed by the model: {len(vetoed)} (compare with the hypothetical "
          f"outcomes of those signals in signals.db to score the model)")
    tokens = sum((r["input_tokens"] or 0) + (r["output_tokens"] or 0) for r in rows)
    print(f"  Tokens used (decisions): {tokens:,}")


def main():
    p = argparse.ArgumentParser(description="Sentinel - DEMO supervisor for gold_monitor signals")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--run", action="store_true", help="run the loop")
    g.add_argument("--once", action="store_true", help="run one cycle and exit")
    g.add_argument("--markets", metavar="TERM", help="search Capital.com demo markets (find epics)")
    g.add_argument("--report", action="store_true", help="summarize decisions.db")
    p.add_argument("--dry-run", action="store_true", help="log decisions, place no orders")
    p.add_argument("--no-llm", action="store_true", help="deterministic pass-through (no model)")
    p.add_argument("--brain", choices=["agent_sdk", "api"],
                   help="agent_sdk = Claude subscription via Claude Code (default); api = Anthropic API key")
    args = p.parse_args()

    setup_logging()
    cfg = SentinelConfig()
    if args.dry_run:
        cfg.dry_run = True
    if args.brain:
        cfg.brain = args.brain

    if args.report:
        cmd_report(cfg)
        return
    try:
        if args.markets:
            from sentinel.broker import DemoBroker
            for m in DemoBroker(cfg).search(args.markets)[:15]:
                print(f"  {m.get('epic', ''):<16} {m.get('instrumentName', '')}")
            return
        sentinel = build(cfg, args.no_llm)
    except RuntimeError as e:
        # Missing credentials / wrong mode / login failure: a clear message,
        # not a traceback (the rules are the message).
        print(f"\n  ERROR: {e}\n")
        sys.exit(1)

    if args.once:
        sentinel.run_once()
    else:
        sentinel.run_forever()


if __name__ == "__main__":
    main()
