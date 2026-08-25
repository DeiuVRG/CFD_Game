import time
import logging
from typing import List

from broker.capital_client import CapitalClient, CapitalAPIError
from broker.session_manager import SessionManager
from config.settings import TRADING, RISK
from data.market_data import MarketDataFetcher
from engine.signal import ApprovedTrade
from monitoring.dashboard import Dashboard
from monitoring.performance_tracker import PerformanceTracker
from monitoring.trade_logger import TradeLogger
from risk.risk_manager import RiskManager
from strategies.base_strategy import BaseStrategy

logger = logging.getLogger(__name__)


class TradingEngine:
    def __init__(
        self,
        client: CapitalClient,
        session_manager: SessionManager,
        market_data: MarketDataFetcher,
        strategies: List[BaseStrategy],
        risk_manager: RiskManager,
        trade_logger: TradeLogger,
        performance_tracker: PerformanceTracker,
        mode: str = "demo",
    ):
        self.client = client
        self.session_manager = session_manager
        self.market_data = market_data
        self.strategies = strategies
        self.risk_manager = risk_manager
        self.trade_logger = trade_logger
        self.tracker = performance_tracker
        self.mode = mode
        self.is_running = False

        # Cache market info to avoid repeated API calls
        self._market_info_cache = {}

    def _get_market_info(self, epic: str):
        if epic not in self._market_info_cache:
            info = self.client.get_market_info(epic)
            if info:
                self._market_info_cache[epic] = info
        return self._market_info_cache.get(epic)

    def _execute_trade(self, trade: ApprovedTrade):
        try:
            confirmation = self.client.create_position(
                epic=trade.signal.epic,
                direction=trade.signal.direction,
                size=trade.size,
                stop_loss=trade.signal.stop_loss,
                take_profit=trade.signal.take_profit,
            )

            if confirmation.status in ("ACCEPTED", "OPEN"):
                self.trade_logger.log_trade(trade, confirmation)
                self.tracker.record_entry(
                    epic=trade.signal.epic,
                    direction=trade.signal.direction,
                    size=trade.size,
                    entry_price=confirmation.level or trade.signal.entry_price,
                    strategy=trade.signal.strategy_name,
                    deal_id=confirmation.deal_id,
                )
                logger.info(
                    f"Trade EXECUTED: {trade.signal.direction} {trade.signal.epic} "
                    f"size={trade.size} deal_id={confirmation.deal_id}"
                )
            else:
                logger.warning(
                    f"Trade REJECTED by broker: {confirmation.status} "
                    f"reason={confirmation.reason}"
                )

        except CapitalAPIError as e:
            logger.error(f"Trade execution failed: {e}")

    def _scan_cycle(self):
        """One full scan: fetch data, analyze, risk check, execute."""
        # Get current equity
        equity = self.client.get_account_equity()
        self.tracker.update_equity(equity)
        self.risk_manager.update_day_start(equity)

        if equity <= 0:
            logger.error("Equity is 0 or negative!")
            self.risk_manager.kill()
            return

        # Check daily loss
        if self.risk_manager.is_daily_limit_hit(equity):
            return

        # Get open positions
        positions = self.client.get_positions()

        # Scan each symbol with each strategy
        signals_found = 0
        for symbol in TRADING.SYMBOLS:
            market_info = self._get_market_info(symbol)
            if not market_info:
                logger.warning(f"Cannot get market info for {symbol}, skipping")
                continue

            if not market_info.market_open:
                continue

            for strategy in self.strategies:
                try:
                    df = self.market_data.get_candles(
                        epic=symbol,
                        resolution=strategy.timeframe,
                        count=max(strategy.get_required_candles(), TRADING.CANDLE_LOOKBACK),
                    )

                    if df.empty or len(df) < strategy.get_required_candles():
                        continue

                    signal = strategy.analyze(symbol, df)
                    if signal is None:
                        continue

                    signals_found += 1

                    approved = self.risk_manager.evaluate_signal(
                        signal, equity, positions, market_info,
                    )
                    if approved is None:
                        continue

                    self._execute_trade(approved)

                    # Refresh positions after trade
                    positions = self.client.get_positions()

                except CapitalAPIError as e:
                    logger.error(f"Error scanning {symbol}/{strategy.name}: {e}")
                except Exception as e:
                    logger.error(
                        f"Unexpected error scanning {symbol}/{strategy.name}: {e}",
                        exc_info=True,
                    )

        # Display dashboard
        summary = self.tracker.get_summary(len(positions))
        strategy_names = [s.name for s in self.strategies]
        Dashboard.print_status(
            summary, positions, strategy_names,
            session_ok=self.session_manager.is_active,
            mode=self.mode,
        )

        if signals_found > 0:
            logger.info(f"Scan complete: {signals_found} signal(s) found")

    def run(self):
        self.is_running = True
        logger.info(f"Trading engine starting ({self.mode} mode)...")
        logger.info(f"Symbols: {TRADING.SYMBOLS}")
        logger.info(f"Strategies: {[s.name for s in self.strategies]}")
        logger.info(f"Scan interval: {TRADING.SCAN_INTERVAL_SEC}s")

        if not self.session_manager.login():
            logger.error("Failed to log in. Exiting.")
            return

        # Initial market info cache
        for symbol in TRADING.SYMBOLS:
            self._get_market_info(symbol)

        while self.is_running:
            try:
                if not self.session_manager.ensure_session():
                    logger.error("Cannot establish session, retrying in 30s...")
                    time.sleep(30)
                    continue

                cycle_start = time.time()
                self._scan_cycle()
                elapsed = time.time() - cycle_start

                sleep_time = max(0, TRADING.SCAN_INTERVAL_SEC - elapsed)
                if sleep_time > 0:
                    time.sleep(sleep_time)

            except KeyboardInterrupt:
                logger.info("Keyboard interrupt received")
                break
            except Exception as e:
                logger.error(f"Error in main loop: {e}", exc_info=True)
                time.sleep(5)

        self.shutdown()

    def shutdown(self):
        logger.info("Shutting down trading engine...")
        self.is_running = False

        summary = self.tracker.get_summary()
        logger.info(
            f"Final stats: Trades={summary.total_trades} "
            f"Wins={summary.wins} Losses={summary.losses} "
            f"Win Rate={summary.win_rate:.0f}% "
            f"P&L={summary.total_pnl:.2f}"
        )

        self.session_manager.logout()
        logger.info("Trading engine stopped.")
