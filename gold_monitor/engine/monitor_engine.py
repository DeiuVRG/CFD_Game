import csv
import logging
import os
import sys
import time
import threading
from datetime import datetime, timezone
from typing import Optional

import pandas as pd

from ai.feature_engineer import FeatureEngineer
from ai.model import GoldPredictor
from config.settings import (INSTRUMENTS, MONITOR, STRATEGY, AI, COSTS,
                             InstrumentConfig, parse_interval_hours)
from data.candles import drop_incomplete_candle
from data.gold_fetcher import MarketFetcher, fetch_all_prices_batch
from data.indicators import Indicators
from data.signal_store import SignalStore, model_version
from engine.signal import Signal
from engine.position_tracker import PositionTracker
from notifications.discord_notify import DiscordNotifier
from strategies.ai_strategy import AIStrategy
from strategies.scalping_strategy import ScalpingStrategy
from strategies.momentum_strategy import MomentumStrategy
from ui.terminal_ui import TerminalUI

logger = logging.getLogger(__name__)


class InstrumentMonitor:
    """Monitors a single instrument with its own fetcher, strategies, and state.

    v3 execution model note: the AI path runs on TRAIN_INTERVAL candles (1h,
    fetched separately over AI_HISTORY_PERIOD) - the same timeframe the model
    was trained and backtested on. The 5m candles are only used for display
    and for the scalping/momentum strategies.
    """

    def __init__(self, instrument: InstrumentConfig):
        self.instrument = instrument
        self.fetcher = MarketFetcher(instrument)
        self.ai_strategy = AIStrategy(model_path=instrument.MODEL_PATH,
                                      instrument=instrument)
        self.scalping = ScalpingStrategy()
        self.momentum = MomentumStrategy()

        self.last_signal: Optional[Signal] = None
        self.last_ai_confidence: float = 0.0
        self.last_ai_probs: dict = {}
        self.last_analysis_time: float = 0

        # AI candles cache (TRAIN_INTERVAL timeframe; refreshed periodically)
        self._ai_df: Optional[pd.DataFrame] = None
        self._ai_df_time: float = 0
        # Open time of the last COMPLETED AI candle processed (ai_only mode)
        self.last_ai_candle_ts = None

        # Display values
        self.price: float = 0
        self.prev_close: float = 0
        self.change: float = 0
        self.change_pct: float = 0
        self.rsi: float = 50.0
        self.macd_hist: float = 0.0
        self.adx: float = 0.0        # 5m ADX (display + scalp/momentum regime)
        self.adx_1h: float = 0.0     # TRAIN_INTERVAL ADX (AI regime gate)
        self.atr: float = 0.0
        self.session_active: bool = True

    @property
    def display_name(self) -> str:
        return self.instrument.SYMBOL_DISPLAY

    @property
    def ai_ready(self) -> bool:
        return self.ai_strategy.predictor.is_ready

    def get_ai_candles(self) -> pd.DataFrame:
        """COMPLETED TRAIN_INTERVAL candles for the AI path (the forming
        candle is dropped). Cached; refreshed after AI_CANDLE_REFRESH_SEC or
        as soon as the clock crosses into a new candle interval, so a new
        completed candle is picked up on the next analysis tick."""
        now = time.time()
        interval_sec = parse_interval_hours(self.instrument.TRAIN_INTERVAL) * 3600
        same_interval = int(now // interval_sec) == int(self._ai_df_time // interval_sec)
        if (self._ai_df is not None and same_interval
                and now - self._ai_df_time < MONITOR.AI_CANDLE_REFRESH_SEC):
            return self._ai_df

        df = self.fetcher.get_candles(
            period=self.instrument.AI_HISTORY_PERIOD,
            interval=self.instrument.TRAIN_INTERVAL,
        )
        if not df.empty:
            self._ai_df = drop_incomplete_candle(df, self.instrument.TRAIN_INTERVAL)
            self._ai_df_time = now

        return self._ai_df if self._ai_df is not None else pd.DataFrame()


class MonitorEngine:
    """Main loop with session filter, regime detection, trailing SL, and EOD close."""

    RETRAIN_INTERVAL_SEC = 6 * 60 * 60  # 6 hours

    def __init__(self):
        self.discord = DiscordNotifier()
        try:
            self.signal_store = SignalStore()
        except Exception as e:
            logger.error(f"Signal store unavailable: {e}")
            self.signal_store = None
        self.position_tracker = PositionTracker(store=self.signal_store)
        self._model_versions: dict[str, str] = {}
        self.is_running = False
        self._pending_command: Optional[str] = None
        self._last_retrain_time: float = 0
        self._retrain_in_progress = False
        self.last_retrain_status: str = ""

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
                    "adx", "rsi", "session",
                ])

    def _model_version(self, mon: InstrumentMonitor) -> str:
        path = mon.instrument.MODEL_PATH
        if path not in self._model_versions:
            self._model_versions[path] = model_version(path)
        return self._model_versions[path]

    def _log_signal(self, mon: InstrumentMonitor, signal: Signal,
                    confidence: float, probs: dict, adx: float, rsi: float,
                    session: bool):
        """Persist a signal: CSV (legacy, human-friendly) + SQLite
        (append-only, with model version). Returns the SQLite row id."""
        instrument = mon.display_name
        with open(self._signal_log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.now(timezone.utc).isoformat(), instrument, signal.direction,
                signal.entry_price, signal.stop_loss, signal.take_profit,
                signal.strategy_name, round(confidence, 3),
                probs.get("BUY", 0), probs.get("SELL", 0),
                round(adx, 1), round(rsi, 1), session,
            ])

        if self.signal_store is None:
            return None
        try:
            return self.signal_store.insert_signal(
                instrument=instrument,
                direction=signal.direction,
                confidence=round(confidence, 4) if confidence else None,
                probabilities=probs,
                adx=round(mon.adx_1h or adx, 2),
                regime=self._get_market_regime(mon.adx_1h or adx),
                entry_price=signal.entry_price,
                stop_loss=signal.stop_loss,
                take_profit=signal.take_profit,
                strategy=signal.strategy_name,
                model_ver=self._model_version(mon),
            )
        except Exception as e:
            logger.error(f"Failed to persist signal: {e}")
            return None

    @staticmethod
    def _is_session_active(instrument: InstrumentConfig = None,
                           now: datetime = None) -> bool:
        """Per-instrument session check.

        24/7 markets (crypto) are always active. Gold/FX keep the classic
        window: weekdays within the London+NY overlap. `now` is injectable
        for tests.
        """
        if instrument is not None and instrument.SESSION_24_7:
            return True

        now = now or datetime.now(timezone.utc)
        hour = now.hour
        # Weekend check: Saturday=5, Sunday=6
        if now.weekday() >= 5:
            return False
        return STRATEGY.SESSION_START_UTC <= hour < STRATEGY.SESSION_END_UTC

    @staticmethod
    def _get_market_regime(adx: float) -> str:
        """Determine market regime from ADX."""
        if adx >= STRATEGY.REGIME_ADX_TRENDING:
            return "TRENDING"
        elif adx >= STRATEGY.REGIME_ADX_THRESHOLD:
            return "MODERATE"
        else:
            return "SIDEWAYS"

    def _vote_signals(
        self,
        ai_signal: Optional[Signal],
        scalp_signal: Optional[Signal],
        mom_signal: Optional[Signal],
        regime: str = "MODERATE",
    ) -> Optional[Signal]:
        """Combine signals with regime-aware voting."""
        buy_score = 0.0
        sell_score = 0.0
        best_signal = None
        best_strength = 0.0

        # Adjust weights based on regime
        if regime == "TRENDING":
            # In trends: momentum matters more, scalping less
            ai_w = STRATEGY.AI_WEIGHT
            scalp_w = STRATEGY.SCALP_WEIGHT * 0.5
            mom_w = STRATEGY.MOMENTUM_WEIGHT * 1.5
        elif regime == "SIDEWAYS":
            # In sideways: scalping matters more, momentum less
            ai_w = STRATEGY.AI_WEIGHT
            scalp_w = STRATEGY.SCALP_WEIGHT * 1.5
            mom_w = STRATEGY.MOMENTUM_WEIGHT * 0.5
        else:
            ai_w = STRATEGY.AI_WEIGHT
            scalp_w = STRATEGY.SCALP_WEIGHT
            mom_w = STRATEGY.MOMENTUM_WEIGHT

        signals = [
            (ai_signal, ai_w),
            (scalp_signal, scalp_w),
            (mom_signal, mom_w),
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

        # Check R:R ratio after costs
        if best_signal.entry_price > 0:
            risk = abs(best_signal.entry_price - best_signal.stop_loss)
            reward = abs(best_signal.take_profit - best_signal.entry_price)
            if risk > 0:
                rr = reward / risk
                if rr < 1.5:
                    return None  # Not worth the risk after costs

        if buy_score > sell_score and buy_score >= STRATEGY.VOTE_THRESHOLD:
            combined_name = " + ".join(contributing)
            return Signal(
                epic=best_signal.epic, direction="BUY",
                entry_price=best_signal.entry_price,
                stop_loss=best_signal.stop_loss,
                take_profit=best_signal.take_profit,
                strategy_name=combined_name, strength=buy_score,
            )

        if sell_score > buy_score and sell_score >= STRATEGY.VOTE_THRESHOLD:
            combined_name = " + ".join(contributing)
            return Signal(
                epic=best_signal.epic, direction="SELL",
                entry_price=best_signal.entry_price,
                stop_loss=best_signal.stop_loss,
                take_profit=best_signal.take_profit,
                strategy_name=combined_name, strength=sell_score,
            )

        return None

    def _run_analysis(self, mon: InstrumentMonitor,
                      df_5m: pd.DataFrame,
                      df_ai: pd.DataFrame) -> Optional[Signal]:
        """v3 execution model: the AI strategy analyzes TRAIN_INTERVAL (1h)
        candles - the timeframe it was trained/backtested on - and its ADX
        regime gate is computed on the same 1h series. Scalping/momentum keep
        using the 5m candles and the 5m ADX regime."""
        epic = mon.display_name
        regime = self._get_market_regime(mon.adx)

        # --- AI path: 1h candles + 1h ADX gate ---
        ai_signal = None
        if not df_ai.empty and len(df_ai) >= 60:
            adx_1h_series = Indicators.adx(
                df_ai["high"], df_ai["low"], df_ai["close"], 14
            )
            adx_1h = adx_1h_series.iloc[-1]
            mon.adx_1h = float(adx_1h) if not pd.isna(adx_1h) else 0.0

            if mon.adx_1h >= mon.instrument.adx_min():
                ai_signal = mon.ai_strategy.analyze(epic, df_ai)

            # Update AI probabilities for display (on the AI timeframe)
            if mon.ai_strategy.predictor.is_ready:
                features = FeatureEngineer.create_features(df_ai)
                if not features.empty and not features.iloc[-1].isna().any():
                    _, mon.last_ai_confidence, mon.last_ai_probs = (
                        mon.ai_strategy.predictor.predict(features)
                    )

        # --- Classic strategies: 5m candles, gated by the 5m ADX regime ---
        scalp_signal = None
        mom_signal = None
        if mon.adx >= STRATEGY.REGIME_ADX_THRESHOLD:
            scalp_signal = mon.scalping.analyze(epic, df_5m)
            mom_signal = mon.momentum.analyze(epic, df_5m)

        combined = self._vote_signals(ai_signal, scalp_signal, mom_signal, regime)
        return combined

    def _process_instrument(self, mon: InstrumentMonitor) -> dict:
        result = {"discord_sent": False, "close_sent": False}

        live = mon.fetcher.get_live_price()
        mon.price = live["price"]
        mon.prev_close = live["prev_close"]
        mon.change = live["change"]
        mon.change_pct = live["change_pct"]

        if mon.price == 0:
            return result

        # Session check (per instrument: crypto is 24/7, gold keeps sessions)
        mon.session_active = self._is_session_active(mon.instrument)

        if MONITOR.SIGNAL_MODE == "ai_only":
            return self._process_ai_only(mon, result)

        # ---- legacy "vote" mode below (not validated by the backtester) ----
        # Check SL/TP with trailing SL (pass ATR for trailing calculation)
        closed_pos = self.position_tracker.check_sl_tp(
            mon.display_name, mon.price, atr=mon.atr
        )
        if closed_pos:
            self.discord.send_close_signal(closed_pos)
            result["close_sent"] = True

        # EOD close check
        eod_closed = self.position_tracker.check_eod_close()
        for pos in eod_closed:
            # Update close price to current price before sending notification
            if pos.close_price == pos.entry_price:
                pos.close_price = mon.price
            self.discord.send_close_signal(pos)
            result["close_sent"] = True

        # Run analysis periodically (only during session)
        now = time.time()
        if now - mon.last_analysis_time >= MONITOR.ANALYSIS_INTERVAL_SEC:
            mon.last_analysis_time = now

            df = mon.fetcher.get_candles()
            if not df.empty and len(df) >= 60:
                close = df["close"]
                high = df["high"]
                low = df["low"]

                rsi_series = Indicators.rsi(close, 14)
                _, _, hist_series = Indicators.macd(close)
                adx_series = Indicators.adx(high, low, close, 14)
                atr_series = Indicators.atr(high, low, close, 14)

                mon.rsi = float(rsi_series.iloc[-1]) if not pd.isna(rsi_series.iloc[-1]) else 50
                mon.macd_hist = float(hist_series.iloc[-1]) if not pd.isna(hist_series.iloc[-1]) else 0
                mon.adx = float(adx_series.iloc[-1]) if not pd.isna(adx_series.iloc[-1]) else 0
                mon.atr = float(atr_series.iloc[-1]) if not pd.isna(atr_series.iloc[-1]) else 0

                # Only generate new signals during an active session.
                # Per-path regime gates: AI is gated by the 1h ADX (inside
                # _run_analysis), scalping/momentum by the 5m ADX.
                if mon.session_active:
                    df_ai = mon.get_ai_candles()
                    signal = self._run_analysis(mon, df, df_ai)

                    if signal:
                        is_new = (
                            mon.last_signal is None or
                            mon.last_signal.direction != signal.direction
                        )
                        old_direction = mon.last_signal.direction if mon.last_signal else None
                        mon.last_signal = signal

                        if is_new:
                            if old_direction and old_direction != signal.direction:
                                existing = self.position_tracker.get_position(mon.display_name)
                                if existing:
                                    closed = self.position_tracker.close_position(
                                        mon.display_name, mon.price, "SIGNAL_REVERSED"
                                    )
                                    if closed:
                                        self.discord.send_close_signal(closed)
                                        result["close_sent"] = True

                            result["discord_sent"] = self.discord.send_signal(
                                signal,
                                confidence=mon.last_ai_confidence,
                                probabilities=mon.last_ai_probs,
                            )
                            signal_id = self._log_signal(
                                mon, signal,
                                mon.last_ai_confidence, mon.last_ai_probs,
                                mon.adx, mon.rsi, mon.session_active,
                            )

                            cost_pct = COSTS.round_trip_cost_pct(
                                mon.instrument, signal.entry_price) * 100
                            self.position_tracker.open_position(
                                instrument=mon.display_name,
                                direction=signal.direction,
                                entry_price=signal.entry_price,
                                stop_loss=signal.stop_loss,
                                take_profit=signal.take_profit,
                                strategy_name=signal.strategy_name,
                                eod_close=not mon.instrument.SESSION_24_7,
                                signal_id=signal_id,
                                cost_pct=cost_pct,
                            )
                    else:
                        mon.last_signal = None

        return result

    # ------------------------------------------------------------------
    # ai_only mode: the live path mirrors the validated v3 backtest
    # ------------------------------------------------------------------

    def _process_ai_only(self, mon: InstrumentMonitor, result: dict) -> dict:
        now = time.time()
        if now - mon.last_analysis_time < MONITOR.ANALYSIS_INTERVAL_SEC:
            return result
        mon.last_analysis_time = now

        # 5m candles: dashboard indicators only (no signal role in this mode)
        df = mon.fetcher.get_candles()
        if not df.empty and len(df) >= 60:
            close, high, low = df["close"], df["high"], df["low"]
            rsi_series = Indicators.rsi(close, 14)
            _, _, hist_series = Indicators.macd(close)
            adx_series = Indicators.adx(high, low, close, 14)
            atr_series = Indicators.atr(high, low, close, 14)
            mon.rsi = float(rsi_series.iloc[-1]) if not pd.isna(rsi_series.iloc[-1]) else 50
            mon.macd_hist = float(hist_series.iloc[-1]) if not pd.isna(hist_series.iloc[-1]) else 0
            mon.adx = float(adx_series.iloc[-1]) if not pd.isna(adx_series.iloc[-1]) else 0
            mon.atr = float(atr_series.iloc[-1]) if not pd.isna(atr_series.iloc[-1]) else 0

        df_ai = mon.get_ai_candles()
        result.update(self._ai_only_step(mon, df_ai, mon.price))
        return result

    def _ai_only_step(self, mon: InstrumentMonitor, df_ai: pd.DataFrame,
                      live_price: float) -> dict:
        """One step of the validated model, run once per NEW completed
        TRAIN_INTERVAL candle:

          1. resolve the open hypothetical position with the v3 exit rules
             on the completed candles after the signal candle;
          2. when flat: XGBoost prediction on the completed candles, gated by
             the TRAIN_INTERVAL ADX, filtered by round-trip cost and min R:R
             exactly like Backtester._simulate_trades. The entry is the live
             price at signal time (~ the open of the forming candle - the
             backtest's "next open"); SL/TP keep the signal-time ATR distances
             anchored to that entry.

        Deliberately absent (not part of the validated model): session
        filter, EOD close, trailing SL, the 3-strategy vote.
        """
        result = {"discord_sent": False, "close_sent": False}
        if df_ai is None or df_ai.empty or len(df_ai) < 60:
            return result
        last_ts = pd.Timestamp(df_ai["timestamp"].iloc[-1])
        if mon.last_ai_candle_ts is not None and last_ts == mon.last_ai_candle_ts:
            return result
        mon.last_ai_candle_ts = last_ts

        name = mon.display_name
        inst = mon.instrument

        # 1) Outcome of the open position on the new completed candle(s)
        closed = self.position_tracker.resolve_with_candles(name, df_ai)
        if closed is not None:
            self.discord.send_close_signal(closed)
            self.discord.reset_last_direction(name)
            result["close_sent"] = True

        # Regime gate on the AI timeframe
        adx_series = Indicators.adx(df_ai["high"], df_ai["low"], df_ai["close"], 14)
        adx = adx_series.iloc[-1]
        mon.adx_1h = float(adx) if not pd.isna(adx) else 0.0
        if pd.isna(adx) or adx < inst.adx_min():
            return result
        if self.position_tracker.get_position(name) is not None:
            return result

        # 2) AI prediction on the completed candles
        sig = mon.ai_strategy.analyze(name, df_ai)
        mon.last_ai_confidence = getattr(mon.ai_strategy, "last_confidence", 0.0)
        mon.last_ai_probs = getattr(mon.ai_strategy, "last_probs", {}) or {}
        if sig is None:
            mon.last_signal = None
            return result

        # Cost + R:R filter, relative to the signal candle close (as backtested)
        close = float(df_ai["close"].iloc[-1])
        sl_dist = abs(sig.entry_price - sig.stop_loss)
        tp_dist = abs(sig.take_profit - sig.entry_price)
        risk = sl_dist / close
        reward = tp_dist / close
        cost = COSTS.round_trip_cost_pct(inst, close)
        if not (risk > 0 and reward > cost and (reward / risk) >= inst.min_rr()):
            logger.info(f"[{name}] AI {sig.direction} filtered: reward={reward:.4f} "
                        f"cost={cost:.4f} rr={reward / risk if risk else 0:.2f}")
            return result

        entry = live_price if live_price and live_price > 0 else close
        if sig.direction == "BUY":
            sl, tp = entry - sl_dist, entry + tp_dist
        else:
            sl, tp = entry + sl_dist, entry - tp_dist
        signal = Signal(
            epic=name, direction=sig.direction, entry_price=entry,
            stop_loss=sl, take_profit=tp, strategy_name=sig.strategy_name,
            strength=sig.strength,
        )
        mon.last_signal = signal

        result["discord_sent"] = self.discord.send_signal(
            signal, confidence=mon.last_ai_confidence,
            probabilities=mon.last_ai_probs,
        )
        signal_id = self._log_signal(
            mon, signal, mon.last_ai_confidence, mon.last_ai_probs,
            mon.adx_1h, mon.rsi, True,
        )
        self.position_tracker.open_position(
            instrument=name, direction=signal.direction, entry_price=entry,
            stop_loss=sl, take_profit=tp, strategy_name=signal.strategy_name,
            eod_close=False, signal_id=signal_id,
            cost_pct=COSTS.round_trip_cost_pct(inst, entry) * 100,
            signal_candle_ts=last_ts,
        )
        return result

    def _batch_update_prices(self):
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
        self._retrain_in_progress = True
        self.last_retrain_status = "Training..."
        logger.info("Background retrain started")

        try:
            for mon in self.monitors:
                inst = mon.instrument
                logger.info(f"Retraining {inst.SYMBOL_DISPLAY}...")

                fetcher = MarketFetcher(inst)
                df = fetcher.get_training_data()
                if df.empty or len(df) < 200:
                    continue

                features = FeatureEngineer.create_features(df)
                labels = FeatureEngineer.create_labels(
                    df, horizon=AI.PREDICTION_HORIZON,
                    threshold=inst.PRICE_CHANGE_THRESHOLD,
                )

                valid_mask = features.notna().all(axis=1) & labels.notna()
                features = features[valid_mask].reset_index(drop=True)
                labels = labels[valid_mask].reset_index(drop=True).astype(int)

                if len(features) < 200:
                    continue

                predictor = GoldPredictor(model_path=inst.MODEL_PATH)
                metrics = predictor.train(features, labels)
                predictor.save()

                mon.ai_strategy.predictor.load()

                acc = metrics["accuracy"] * 100
                cv = metrics["cv_accuracy_mean"] * 100
                logger.info(f"Retrained {inst.SYMBOL_DISPLAY}: test={acc:.1f}% cv={cv:.1f}%")

            now = datetime.now(timezone.utc).strftime("%H:%M")
            self.last_retrain_status = f"OK ({now})"
            self._last_retrain_time = time.time()

        except Exception as e:
            self.last_retrain_status = f"Error: {e}"
            logger.error(f"Retrain error: {e}", exc_info=True)
        finally:
            self._retrain_in_progress = False

    def _maybe_retrain(self):
        if self._retrain_in_progress:
            return
        now = time.time()
        if self._last_retrain_time == 0:
            if now - self._start_time < 300:
                return
        elif now - self._last_retrain_time < self.RETRAIN_INTERVAL_SEC:
            return

        thread = threading.Thread(target=self._retrain_background, daemon=True)
        thread.start()

    @staticmethod
    def _stdin_is_interactive() -> bool:
        try:
            return sys.stdin is not None and sys.stdin.isatty()
        except (AttributeError, ValueError, OSError):
            return False

    def _restore_positions(self, mon: InstrumentMonitor):
        """Restart safety: signals left without an outcome by a previous run
        are replayed against the completed TRAIN_INTERVAL candles with the v3
        rules; the newest unresolved one becomes the open position again."""
        if self.signal_store is None:
            return
        inst = mon.instrument
        try:
            df = mon.get_ai_candles()
            info = self.position_tracker.restore_from_store(
                mon.display_name, df,
                cost_pct_fn=lambda price, inst=inst:
                    COSTS.round_trip_cost_pct(inst, price) * 100,
                interval_hours=parse_interval_hours(inst.TRAIN_INTERVAL),
                eod_close=not inst.SESSION_24_7,
            )
            if info["resolved"] or info["open"]:
                logger.info(f"{mon.display_name}: restored state - "
                            f"{info['resolved']} outcome(s) replayed, "
                            f"open position: {info['open']}")
        except Exception as e:
            logger.error(f"{mon.display_name}: restore failed: {e}", exc_info=True)

    def _keyboard_listener(self):
        if sys.platform == "win32":
            import msvcrt
            while self.is_running:
                if msvcrt.kbhit():
                    key = msvcrt.getch()
                    if key == b'\x17':
                        self._pending_command = "ctrl_w"
                    elif key == b'\x0c':
                        self._pending_command = "ctrl_l"
                time.sleep(0.1)
        else:
            import select
            while self.is_running:
                if select.select([sys.stdin], [], [], 0.1)[0]:
                    key = sys.stdin.read(1)
                    if key == '':
                        # EOF (stdin is /dev/null under systemd, a closed
                        # pipe, ...): without this break select() reports
                        # "readable" forever and the loop spins at 100% CPU.
                        logger.info("stdin closed - keyboard listener stopped")
                        return
                    if key == '\x17':
                        self._pending_command = "ctrl_w"
                    elif key == '\x0c':
                        self._pending_command = "ctrl_l"

    def _process_keyboard(self):
        cmd = self._pending_command
        if cmd is None:
            return
        self._pending_command = None

        open_positions = []
        for mon in self.monitors:
            pos = self.position_tracker.get_position(mon.display_name)
            if pos:
                open_positions.append((mon, pos))

        if cmd == "ctrl_w":
            for mon, pos in open_positions:
                closed = self.position_tracker.close_position(
                    mon.display_name, mon.price, "MANUAL_WIN"
                )
                if closed:
                    self.discord.send_close_signal(closed)
        elif cmd == "ctrl_l":
            for mon, pos in open_positions:
                closed = self.position_tracker.close_position(
                    mon.display_name, mon.price, "MANUAL_LOSS"
                )
                if closed:
                    self.discord.send_close_signal(closed)

    def run(self):
        self.is_running = True
        self._start_time = time.time()
        logger.info(f"Trading Monitor v2 starting with {len(self.monitors)} instruments...")

        # Keyboard shortcuts only make sense on an interactive terminal. Under
        # systemd/cron/nohup stdin is /dev/null and the listener must not run.
        if self._stdin_is_interactive():
            kb_thread = threading.Thread(target=self._keyboard_listener, daemon=True)
            kb_thread.start()
        else:
            logger.info("Non-interactive stdin - keyboard shortcuts disabled")

        for mon in self.monitors:
            live = mon.fetcher.get_live_price()
            mon.price = live["price"]
            mon.prev_close = live["prev_close"]
            mon.change = live["change"]
            mon.change_pct = live["change_pct"]
            self._restore_positions(mon)

        while self.is_running:
            try:
                self._process_keyboard()
                self._maybe_retrain()
                self._batch_update_prices()

                results = {}
                for mon in self.monitors:
                    results[mon.display_name] = self._process_instrument(mon)

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
