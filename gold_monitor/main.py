#!/usr/bin/env python3
"""
Trading Monitor - Real-time multi-instrument monitoring with AI signals + Discord notifications.

Usage:
    python main.py --train              Train AI models for ALL instruments
    python main.py --train gold         Train only Gold model
    python main.py --train eurusd       Train only EUR/USD model
    python main.py --monitor            Start live monitoring (all instruments)
    python main.py --test-discord       Send a test message to Discord
"""

import argparse
import logging
import os
import sys

# Add project root to path
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
    """Train XGBoost AI models."""
    from ai.trainer import train_model
    success = train_model(instrument_filter)
    if not success:
        sys.exit(1)


def cmd_monitor():
    """Start live multi-instrument monitoring."""
    from engine.monitor_engine import MonitorEngine
    from config.settings import INSTRUMENTS

    enabled = [i for i in INSTRUMENTS if i.ENABLED]
    print(f"\n  Starting Trading Monitor...")
    print(f"  Instruments: {', '.join(i.SYMBOL_DISPLAY for i in enabled)}")
    print(f"  Press Ctrl+C to stop\n")

    engine = MonitorEngine()
    try:
        engine.run()
    except KeyboardInterrupt:
        print("\n  Stopped.")


def cmd_test_discord():
    """Send a test message to Discord to verify webhook."""
    from notifications.discord_notify import DiscordNotifier
    from config.settings import DISCORD

    if not DISCORD.WEBHOOK_URL:
        print("ERROR: DISCORD_WEBHOOK_URL not set in .env")
        print("1. Go to Discord -> Server Settings -> Integrations -> Webhooks")
        print("2. Create a webhook and copy the URL")
        print("3. Add it to your .env file")
        sys.exit(1)

    notifier = DiscordNotifier()
    success = notifier.send_test()

    if success:
        print("Test message sent successfully! Check your Discord channel.")
    else:
        print("Failed to send test message. Check your webhook URL.")


def main():
    parser = argparse.ArgumentParser(
        description="Trading Monitor - AI-powered multi-instrument trading signals",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--train", nargs="?", const="all", metavar="INSTRUMENT",
                       help="Train AI model (all, gold, eurusd, gbpusd)")
    group.add_argument("--monitor", action="store_true", help="Start live monitoring")
    group.add_argument("--test-discord", action="store_true", help="Test Discord webhook")

    args = parser.parse_args()
    setup_logging()

    if args.train:
        instrument_filter = None if args.train == "all" else args.train
        cmd_train(instrument_filter)
    elif args.monitor:
        cmd_monitor()
    elif args.test_discord:
        cmd_test_discord()


if __name__ == "__main__":
    main()
