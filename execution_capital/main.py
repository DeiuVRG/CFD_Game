#!/usr/bin/env python3
"""
CFD Trading Bot - Automated trading on Capital.com
"""

import argparse
import logging
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from broker.capital_client import CapitalClient
from broker.session_manager import SessionManager
from config.credentials import load_credentials
from config.settings import TRADING
from data.market_data import MarketDataFetcher
from engine.trading_engine import TradingEngine
from monitoring.performance_tracker import PerformanceTracker
from monitoring.trade_logger import TradeLogger
from risk.risk_manager import RiskManager
from strategies.momentum_strategy import MomentumStrategy
from strategies.mean_reversion_strategy import MeanReversionStrategy
from strategies.scalping_strategy import ScalpingStrategy


def setup_logging(mode: str):
    os.makedirs("logs", exist_ok=True)
    log_format = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"

    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=[
            logging.FileHandler(f"logs/bot_{mode}.log"),
            logging.StreamHandler(sys.stdout),
        ],
    )


def main():
    parser = argparse.ArgumentParser(description="CFD Trading Bot")
    parser.add_argument(
        "--mode", choices=["demo", "live"], default=None,
        help="Trading mode (overrides .env CAPITAL_MODE)",
    )
    parser.add_argument(
        "--symbols", nargs="+", default=None,
        help="Symbols to trade (e.g., EURUSD GBPUSD)",
    )
    args = parser.parse_args()

    # Load credentials
    credentials = load_credentials()
    if args.mode:
        credentials.mode = args.mode
        from config.settings import BROKER
        credentials.base_url = (
            BROKER.LIVE_URL if args.mode == "live" else BROKER.DEMO_URL
        )

    mode = credentials.mode
    setup_logging(mode)
    logger = logging.getLogger(__name__)

    logger.info(f"Starting CFD Trading Bot ({mode} mode)")
    logger.info(f"Base URL: {credentials.base_url}")

    if args.symbols:
        TRADING.SYMBOLS = args.symbols
    logger.info(f"Symbols: {TRADING.SYMBOLS}")

    if mode == "live":
        logger.warning("=" * 50)
        logger.warning("  LIVE TRADING MODE - REAL MONEY AT RISK")
        logger.warning("=" * 50)

    # Initialize components
    client = CapitalClient(credentials.base_url)
    session_manager = SessionManager(client, credentials)
    market_data = MarketDataFetcher(session_manager, client)

    strategies = [
        ScalpingStrategy(),
        MomentumStrategy(),
        MeanReversionStrategy(),
    ]

    risk_manager = RiskManager()
    trade_logger = TradeLogger(output_dir="output")
    performance_tracker = PerformanceTracker()

    engine = TradingEngine(
        client=client,
        session_manager=session_manager,
        market_data=market_data,
        strategies=strategies,
        risk_manager=risk_manager,
        trade_logger=trade_logger,
        performance_tracker=performance_tracker,
        mode=mode,
    )

    try:
        engine.run()
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.critical(f"Fatal error: {e}", exc_info=True)
    finally:
        logger.info("Bot stopped.")


if __name__ == "__main__":
    main()
