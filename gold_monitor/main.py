#!/usr/bin/env python3
"""
Trading Monitor v2 - Real-time multi-instrument monitoring with AI signals.

Usage:
    python main.py --train              Train AI models for ALL instruments
    python main.py --train gold         Train only Gold model
    python main.py --train eurusd       Train only EUR/USD model
    python main.py --monitor            Start live monitoring
    python main.py --backtest           Run backtest with cost modeling
    python main.py --backtest gold      Backtest only Gold
    python main.py --test-discord       Send a test message to Discord
"""

import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def setup_logging():
    os.makedirs("logs", exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.FileHandler("logs/monitor.log", encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )


def cmd_train(instrument_filter: str = None):
    from ai.trainer import train_model
    success = train_model(instrument_filter)
    if not success:
        sys.exit(1)


def cmd_backtest(instrument_filter: str = None):
    """Run backtest with realistic cost modeling and risk-adjusted metrics."""
    from ai.backtester import Backtester
    from config.settings import INSTRUMENTS
    from data.gold_fetcher import MarketFetcher

    targets = [i for i in INSTRUMENTS if i.ENABLED]
    if instrument_filter:
        targets = [i for i in targets if instrument_filter.lower() in i.SYMBOL.lower()
                    or instrument_filter.lower() in i.SYMBOL_DISPLAY.lower()]
        if not targets:
            print(f"ERROR: No instrument matching '{instrument_filter}'")
            sys.exit(1)

    for inst in targets:
        print(f"\nDownloading data for {inst.SYMBOL_DISPLAY}...")
        fetcher = MarketFetcher(inst)
        df = fetcher.get_training_data()

        if df.empty:
            print(f"ERROR: No data for {inst.SYMBOL}")
            continue

        print(f"Running backtest on {len(df)} candles...")
        bt = Backtester(inst)
        result = bt.run(df)
        Backtester.print_report(result, inst.SYMBOL_DISPLAY)


def cmd_backtest_wf(instrument_filter: str = None):
    """Walk-forward backtest: re-trains model periodically (realistic simulation)."""
    from ai.backtester import Backtester
    from config.settings import INSTRUMENTS
    from data.gold_fetcher import MarketFetcher

    targets = [i for i in INSTRUMENTS if i.ENABLED]
    if instrument_filter:
        targets = [i for i in targets if instrument_filter.lower() in i.SYMBOL.lower()
                    or instrument_filter.lower() in i.SYMBOL_DISPLAY.lower()]
        if not targets:
            print(f"ERROR: No instrument matching '{instrument_filter}'")
            sys.exit(1)

    for inst in targets:
        print(f"\nDownloading data for {inst.SYMBOL_DISPLAY}...")
        fetcher = MarketFetcher(inst)
        df = fetcher.get_training_data()

        if df.empty:
            print(f"ERROR: No data for {inst.SYMBOL}")
            continue

        print(f"Running WALK-FORWARD backtest on {len(df)} candles...")
        print(f"  (Re-trains model every 500 candles - simulates real production)")
        bt = Backtester(inst)
        result = bt.run_walk_forward(df, retrain_every=500, min_train=2000)

        # Print using standard report (just OOS section)
        Backtester._print_section(
            result.get("metrics", {}),
            result.get("trades", []),
            f"WALK-FORWARD: {inst.SYMBOL_DISPLAY} ({result.get('n_retrains', 0)} retrains)"
        )

        trades = result.get("trades", [])
        if trades:
            print(f"\n  Last 10 trades:")
            for t in trades[-10:]:
                emoji = "W" if t.pnl_net_pct > 0 else "L"
                print(f"    [{emoji}] {t.direction:4s} @ {t.entry_price:.2f} -> "
                      f"{t.exit_price:.2f} | P&L: {t.pnl_net_pct:+.3f}% "
                      f"(cost: {t.cost_pct:.3f}%) [{t.strategy}]")

        print(f"{'=' * 60}")


def cmd_monitor():
    from engine.monitor_engine import MonitorEngine
    from config.settings import INSTRUMENTS

    enabled = [i for i in INSTRUMENTS if i.ENABLED]
    print(f"\n  Starting Trading Monitor v2...")
    print(f"  Instruments: {', '.join(i.SYMBOL_DISPLAY for i in enabled)}")
    print(f"  Features: Walk-forward AI, Trailing SL, Session filter, Regime detection")
    print(f"  Press Ctrl+C to stop\n")

    engine = MonitorEngine()
    try:
        engine.run()
    except KeyboardInterrupt:
        print("\n  Stopped.")


def cmd_optimize(instrument_filter: str = None):
    """Find optimal parameters via grid search."""
    from ai.optimizer import optimize_instrument
    from config.settings import INSTRUMENTS
    from data.gold_fetcher import MarketFetcher

    targets = [i for i in INSTRUMENTS if i.ENABLED]
    if instrument_filter:
        targets = [i for i in targets if instrument_filter.lower() in i.SYMBOL.lower()
                    or instrument_filter.lower() in i.SYMBOL_DISPLAY.lower()]
        if not targets:
            print(f"ERROR: No instrument matching '{instrument_filter}'")
            sys.exit(1)

    for inst in targets:
        print(f"\nDownloading data for {inst.SYMBOL_DISPLAY}...")
        fetcher = MarketFetcher(inst)
        df = fetcher.get_training_data()

        if df.empty:
            print(f"ERROR: No data for {inst.SYMBOL}")
            continue

        results = optimize_instrument(inst, df)

        if results:
            best = results[0]
            print(f"\n  >>> RECOMMENDED for {inst.SYMBOL_DISPLAY}:")
            print(f"      SL = {best.sl_atr}x ATR")
            print(f"      TP = {best.tp_atr}x ATR")
            print(f"      AI Confidence >= {best.confidence_threshold}")
            print(f"      Min R:R = {best.min_rr}")
            print(f"      ADX filter >= {best.adx_filter}")


def cmd_test_discord():
    from notifications.discord_notify import DiscordNotifier
    from config.settings import DISCORD

    if not DISCORD.WEBHOOK_URL:
        print("ERROR: DISCORD_WEBHOOK_URL not set in .env")
        sys.exit(1)

    notifier = DiscordNotifier()
    success = notifier.send_test()
    print("Test sent!" if success else "Failed!")


def main():
    parser = argparse.ArgumentParser(
        description="Trading Monitor v2 - AI signals + Discord",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--train", nargs="?", const="all", metavar="INSTRUMENT",
                       help="Train AI model (all, gold, eurusd, gbpusd)")
    group.add_argument("--monitor", action="store_true", help="Start live monitoring")
    group.add_argument("--backtest", nargs="?", const="all", metavar="INSTRUMENT",
                       help="Run backtest with cost modeling (fixed split)")
    group.add_argument("--backtest-wf", nargs="?", const="all", metavar="INSTRUMENT",
                       help="Walk-forward backtest (re-trains periodically)")
    group.add_argument("--optimize", nargs="?", const="all", metavar="INSTRUMENT",
                       help="Find optimal SL/TP/confidence parameters")
    group.add_argument("--test-discord", action="store_true", help="Test Discord webhook")

    args = parser.parse_args()
    setup_logging()

    if args.train:
        instrument_filter = None if args.train == "all" else args.train
        cmd_train(instrument_filter)
    elif args.monitor:
        cmd_monitor()
    elif args.backtest is not None:
        instrument_filter = None if args.backtest == "all" else args.backtest
        cmd_backtest(instrument_filter)
    elif getattr(args, 'backtest_wf', None) is not None:
        instrument_filter = None if args.backtest_wf == "all" else args.backtest_wf
        cmd_backtest_wf(instrument_filter)
    elif args.optimize is not None:
        instrument_filter = None if args.optimize == "all" else args.optimize
        cmd_optimize(instrument_filter)
    elif args.test_discord:
        cmd_test_discord()


if __name__ == "__main__":
    main()
