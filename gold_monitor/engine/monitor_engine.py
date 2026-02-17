import csv
import logging
import os
import sys
import time
import threading
from datetime import datetime
from typing import Optional

import pandas as pd

from ai.feature_engineer import FeatureEngineer
from ai.model import GoldPredictor
from config.settings import INSTRUMENTS, MONITOR, STRATEGY, AI, InstrumentConfig
from data.gold_fetcher import MarketFetcher, fetch_all_prices_batch
from data.indicators import Indicators
from engine.signal import Signal
from engine.position_tracker import PositionTracker
from notifications.discord_notify import DiscordNotifier
from strategies.ai_strategy import AIStrategy
from strategies.scalping_strategy import ScalpingStrategy
from strategies.momentum_strategy import MomentumStrategy
from ui.terminal_ui import TerminalUI

logger = logging.getLogger(__name__)


class InstrumentMonitor:
    """Monitors a single instrument with its own fetcher, strategies, and state."""

    def __init__(self, instrument: InstrumentConfig):
        self.instrument = instrument
        self.fetcher = MarketFetcher(instrument)
        self.ai_strategy = AIStrategy(model_path=instrument.MODEL_PATH)
        self.scalping = ScalpingStrategy()
        self.momentum = MomentumStrategy()

        self.last_signal: Optional[Signal] = None
        self.last_ai_confidence: float = 0.0
        self.last_ai_probs: dict = {}
        self.last_analysis_time: float = 0

        # Display values
        self.price: float = 0
        self.prev_close: float = 0
        self.change: float = 0
        self.change_pct: float = 0
        self.rsi: float = 50.0
        self.macd_hist: float = 0.0

    @property
    def display_name(self) -> str:
        return self.instrument.SYMBOL_DISPLAY

    @property
    def ai_ready(self) -> bool:
        return self.ai_strategy.predictor.is_ready


class MonitorEngine:
    """Main loop: monitors multiple instruments, tracks positions, notifies Discord."""

    # Re-train every 6 hours
    RETRAIN_INTERVAL_SEC = 6 * 60 * 60

    def __init__(self):
        self.discord = DiscordNotifier()
        self.position_tracker = PositionTracker()
        self.is_running = False
        self._pending_command: Optional[str] = None
        self._last_retrain_time: float = 0
        self._retrain_in_progress = False
        self.last_retrain_status: str = ""

        # Create a monitor for each enabled instrument
        self.monitors: list[InstrumentMonitor] = []
        for inst in INSTRUMENTS:
            if inst.ENABLED:
                self.monitors.append(InstrumentMonitor(inst))
                logger.info(f"Loaded instrument: {inst.SYMBOL_DISPLAY}")

        self._signal_log_path = "output/signals.csv"
        os.makedirs("output", exist_ok=True)
        self._init_signal_log()

    def _init_signal_log(self):
        if not os.path.exists(self._signal_log_path):
            with open(self._signal_log_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "timestamp", "instrument", "direction", "price", "stop_loss",
                    "take_profit", "strategy", "confidence", "ai_buy_prob", "ai_sell_prob",
                ])

    def _log_signal(self, instrument: str, signal: Signal, confidence: float, probs: dict):
        with open(self._signal_log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.utcnow().isoformat(),
                instrument,
                signal.direction,
                signal.entry_price,
                signal.stop_loss,
                signal.take_profit,
                signal.strategy_name,
                round(confidence, 3),
                probs.get("BUY", 0),
                probs.get("SELL", 0),
            ])

    def _vote_signals(
        self,
        ai_signal: Optional[Signal],
        scalp_signal: Optional[Signal],
        mom_signal: Optional[Signal],
    ) -> Optional[Signal]:
        """Combine signals from all strategies using weighted voting."""
        buy_score = 0.0
        sell_score = 0.0
        best_signal = None
        best_strength = 0.0

        signals = [
            (ai_signal, STRATEGY.AI_WEIGHT),
            (scalp_signal, STRATEGY.SCALP_WEIGHT),
            (mom_signal, STRATEGY.MOMENTUM_WEIGHT),
        ]

        contributing = []
        for sig, weight in signals:
            if sig is None:
                continue
            if sig.direction == "BUY":
                buy_score += weight * sig.strength
                contributing.append(sig.strategy_name)
                if sig.strength > best_strength:
                    best_signal = sig
                    best_strength = sig.strength
            elif sig.direction == "SELL":
                sell_score += weight * sig.strength
                contributing.append(sig.strategy_name)
                if sig.strength > best_strength:
                    best_signal = sig
                    best_strength = sig.strength

        if best_signal is None:
            return None

        if buy_score > sell_score and buy_score >= STRATEGY.VOTE_THRESHOLD:
            combined_name = " + ".join(contributing)
            return Signal(
                epic=best_signal.epic,
                direction="BUY",
                entry_price=best_signal.entry_price,
                stop_loss=best_signal.stop_loss,
                take_profit=best_signal.take_profit,
                strategy_name=combined_name,
                strength=buy_score,
            )

        if sell_score > buy_score and sell_score >= STRATEGY.VOTE_THRESHOLD:
            combined_name = " + ".join(contributing)
            return Signal(
                epic=best_signal.epic,
                direction="SELL",
                entry_price=best_signal.entry_price,
                stop_loss=best_signal.stop_loss,
                take_profit=best_signal.take_profit,
                strategy_name=combined_name,
                strength=sell_score,
            )

        return None

    def _run_analysis(self, mon: InstrumentMonitor, df: pd.DataFrame) -> Optional[Signal]:
        """Run all strategies for a single instrument and combine via voting."""
        epic = mon.display_name

        ai_signal = mon.ai_strategy.analyze(epic, df)
        scalp_signal = mon.scalping.analyze(epic, df)
        mom_signal = mon.momentum.analyze(epic, df)

        # Update AI probabilities for display
        if mon.ai_strategy.predictor.is_ready:
            features = FeatureEngineer.create_features(df)
            if not features.empty and not features.iloc[-1].isna().any():
                _, mon.last_ai_confidence, mon.last_ai_probs = (
                    mon.ai_strategy.predictor.predict(features)
                )

        combined = self._vote_signals(ai_signal, scalp_signal, mom_signal)
        return combined

    def _process_instrument(self, mon: InstrumentMonitor) -> dict:
        """Process a single instrument: fetch price, check SL/TP, run analysis."""
        result = {"discord_sent": False, "close_sent": False}

        # Fetch live price
        live = mon.fetcher.get_live_price()
        mon.price = live["price"]
        mon.prev_close = live["prev_close"]
        mon.change = live["change"]
        mon.change_pct = live["change_pct"]

        if mon.price == 0:
            logger.warning(f"Could not get price for {mon.display_name}")
            return result

        # Check SL/TP for open positions
        closed_pos = self.position_tracker.check_sl_tp(mon.display_name, mon.price)
        if closed_pos:
            self.discord.send_close_signal(closed_pos)
            result["close_sent"] = True
            logger.info(f"Position closed for {mon.display_name}: {closed_pos.close_reason}")

        # Run analysis periodically
        now = time.time()
        if now - mon.last_analysis_time >= MONITOR.ANALYSIS_INTERVAL_SEC:
            mon.last_analysis_time = now

            df = mon.fetcher.get_candles()
            if not df.empty and len(df) >= 60:
                # Calculate display indicators
                close = df["close"]
                rsi_series = Indicators.rsi(close, 14)
                _, _, hist_series = Indicators.macd(close)

                mon.rsi = rsi_series.iloc[-1] if not pd.isna(rsi_series.iloc[-1]) else 50
                mon.macd_hist = hist_series.iloc[-1] if not pd.isna(hist_series.iloc[-1]) else 0

                # Run strategies
                signal = self._run_analysis(mon, df)

                if signal:
                    is_new = (
                        mon.last_signal is None or
                        mon.last_signal.direction != signal.direction
                    )
                    old_direction = mon.last_signal.direction if mon.last_signal else None
                    mon.last_signal = signal

                    if is_new:
                        # If signal reversed and we have an open position, close it
                        if old_direction and old_direction != signal.direction:
                            existing = self.position_tracker.get_position(mon.display_name)
                            if existing:
                                closed = self.position_tracker.close_position(
                                    mon.display_name, mon.price, "SIGNAL_REVERSED"
                                )
                                if closed:
                                    self.discord.send_close_signal(closed)
                                    result["close_sent"] = True

                        # Send new signal
                        result["discord_sent"] = self.discord.send_signal(
                            signal,
                            confidence=mon.last_ai_confidence,
                            probabilities=mon.last_ai_probs,
                        )
                        self._log_signal(
                            mon.display_name, signal,
                            mon.last_ai_confidence, mon.last_ai_probs,
                        )

                        # Track the new position
                        self.position_tracker.open_position(
                            instrument=mon.display_name,
                            direction=signal.direction,
                            entry_price=signal.entry_price,
                            stop_loss=signal.stop_loss,
                            take_profit=signal.take_profit,
                            strategy_name=signal.strategy_name,
                        )
                else:
                    mon.last_signal = None

        return result

    def _batch_update_prices(self):
        """Fetch all prices in 1 API call and update monitors."""
        enabled_instruments = [mon.instrument for mon in self.monitors]
        prices = fetch_all_prices_batch(enabled_instruments)

        for mon in self.monitors:
            td_sym = mon.instrument.TWELVEDATA_SYMBOL
            if td_sym in prices:
                mon.fetcher.update_from_batch(prices[td_sym])
                live = mon.fetcher.get_live_price()
                mon.price = live["price"]
                mon.prev_close = live["prev_close"]
                mon.change = live["change"]
                mon.change_pct = live["change_pct"]

    def _retrain_background(self):
        """Re-train all AI models in a background thread."""
        self._retrain_in_progress = True
        self.last_retrain_status = "Training..."
        logger.info("Background retrain started")

        try:
            for mon in self.monitors:
                inst = mon.instrument
                logger.info(f"Retraining {inst.SYMBOL_DISPLAY}...")

                # Download fresh data
                fetcher = MarketFetcher(inst)
                df = fetcher.get_training_data()
                if df.empty or len(df) < 200:
                    logger.warning(f"Not enough data for {inst.SYMBOL_DISPLAY}")
                    continue

                # Create features and labels
                features = FeatureEngineer.create_features(df)
                labels = FeatureEngineer.create_labels(
                    df, horizon=AI.PREDICTION_HORIZON,
                    threshold=inst.PRICE_CHANGE_THRESHOLD,
                )

                valid_mask = features.notna().all(axis=1) & labels.notna()
                features = features[valid_mask].reset_index(drop=True)
                labels = labels[valid_mask].reset_index(drop=True)

                if len(features) < 100:
                    continue

                # Train new model
                predictor = GoldPredictor(model_path=inst.MODEL_PATH)
                metrics = predictor.train(features, labels)
                predictor.save()

                # Hot-reload into the running monitor
                mon.ai_strategy.predictor.load()

                acc = metrics["accuracy"] * 100
                logger.info(f"Retrained {inst.SYMBOL_DISPLAY}: {acc:.1f}% accuracy")

            now = datetime.utcnow().strftime("%H:%M")
            self.last_retrain_status = f"OK ({now})"
            self._last_retrain_time = time.time()
            logger.info("Background retrain completed")

        except Exception as e:
            self.last_retrain_status = f"Error: {e}"
            logger.error(f"Retrain error: {e}", exc_info=True)
        finally:
            self._retrain_in_progress = False

    def _maybe_retrain(self):
        """Start retrain if enough time has passed."""
        if self._retrain_in_progress:
            return
        now = time.time()
        if self._last_retrain_time == 0:
            # First retrain: wait 5 minutes after startup
            if now - self._start_time < 300:
                return
        elif now - self._last_retrain_time < self.RETRAIN_INTERVAL_SEC:
            return

        thread = threading.Thread(target=self._retrain_background, daemon=True)
        thread.start()

    def _keyboard_listener(self):
        """Listen for keyboard input in a background thread.
        Ctrl+W = chr(23), Ctrl+L = chr(12)
        """
        if sys.platform == "win32":
            import msvcrt
            while self.is_running:
                if msvcrt.kbhit():
                    key = msvcrt.getch()
                    if key == b'\x17':      # Ctrl+W
                        self._pending_command = "ctrl_w"
                    elif key == b'\x0c':    # Ctrl+L
                        self._pending_command = "ctrl_l"
                time.sleep(0.1)
        else:
            import select
            while self.is_running:
                if select.select([sys.stdin], [], [], 0.1)[0]:
                    key = sys.stdin.read(1)
                    if key == '\x17':
                        self._pending_command = "ctrl_w"
                    elif key == '\x0c':
                        self._pending_command = "ctrl_l"

    def _process_keyboard(self):
        """Process pending keyboard commands."""
        cmd = self._pending_command
        if cmd is None:
            return

        self._pending_command = None

        # Find instruments with open positions
        open_positions = []
        for mon in self.monitors:
            pos = self.position_tracker.get_position(mon.display_name)
            if pos:
                open_positions.append((mon, pos))

        if cmd == "ctrl_w":
            # Ctrl+W = Close on WIN
            for mon, pos in open_positions:
                closed = self.position_tracker.close_position(
                    mon.display_name, mon.price, "MANUAL_WIN"
                )
                if closed:
                    self.discord.send_close_signal(closed)
                    logger.info(f"Manual WIN close: {mon.display_name} @ {mon.price}")

        elif cmd == "ctrl_l":
            # Ctrl+L = Close on LOSS
            for mon, pos in open_positions:
                closed = self.position_tracker.close_position(
                    mon.display_name, mon.price, "MANUAL_LOSS"
                )
                if closed:
                    self.discord.send_close_signal(closed)
                    logger.info(f"Manual LOSS close: {mon.display_name} @ {mon.price}")

    def run(self):
        """Main monitoring loop for all instruments."""
        self.is_running = True
        self._start_time = time.time()
        logger.info(f"Trading Monitor starting with {len(self.monitors)} instruments...")

        # Start keyboard listener thread
        kb_thread = threading.Thread(target=self._keyboard_listener, daemon=True)
        kb_thread.start()

        # Initial fetch with prev_close (uses /quote endpoint, 1 per instrument)
        for mon in self.monitors:
            live = mon.fetcher.get_live_price()
            mon.price = live["price"]
            mon.prev_close = live["prev_close"]
            mon.change = live["change"]
            mon.change_pct = live["change_pct"]

        while self.is_running:
            try:
                # Process keyboard commands
                self._process_keyboard()

                # Auto-retrain AI models periodically
                self._maybe_retrain()

                # Single batch API call for all prices
                self._batch_update_prices()

                results = {}
                for mon in self.monitors:
                    results[mon.display_name] = self._process_instrument(mon)

                # Display UI
                TerminalUI.display_multi(
                    monitors=self.monitors,
                    position_tracker=self.position_tracker,
                    results=results,
                    retrain_status=self.last_retrain_status,
                )

                time.sleep(MONITOR.FETCH_INTERVAL_SEC)

            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.error(f"Monitor error: {e}", exc_info=True)
                time.sleep(5)

        logger.info("Trading Monitor stopped.")
