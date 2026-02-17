import logging
from datetime import date
from typing import Optional, List

from broker.models import OpenPosition, MarketInfo
from config.settings import RISK
from engine.signal import Signal, ApprovedTrade
from risk.position_sizer import PositionSizer

logger = logging.getLogger(__name__)


class RiskManager:
    def __init__(self):
        self.is_active: bool = True
        self._day_start_equity: float = 0.0
        self._current_day: date = date.today()
        self._trades_today: int = 0

    def update_day_start(self, equity: float):
        today = date.today()
        if today != self._current_day:
            self._current_day = today
            self._day_start_equity = equity
            self._trades_today = 0
            logger.info(f"New trading day. Start equity: {equity:.2f}")
        elif self._day_start_equity == 0.0:
            self._day_start_equity = equity

    def is_daily_limit_hit(self, current_equity: float) -> bool:
        if self._day_start_equity <= 0:
            return False
        loss_pct = (self._day_start_equity - current_equity) / self._day_start_equity
        if loss_pct >= RISK.MAX_DAILY_LOSS:
            logger.warning(
                f"Daily loss limit hit: {loss_pct*100:.1f}% "
                f"(limit: {RISK.MAX_DAILY_LOSS*100}%)"
            )
            return True
        return False

    def evaluate_signal(
        self,
        signal: Signal,
        equity: float,
        open_positions: List[OpenPosition],
        market_info: MarketInfo,
    ) -> Optional[ApprovedTrade]:
        if not self.is_active:
            logger.warning("Kill switch active - rejecting signal")
            return None

        # Check daily loss limit
        if self.is_daily_limit_hit(equity):
            return None

        # Check max concurrent positions
        if len(open_positions) >= RISK.MAX_CONCURRENT_POSITIONS:
            logger.info(
                f"Max positions reached ({RISK.MAX_CONCURRENT_POSITIONS}) "
                f"- rejecting {signal.epic} {signal.direction}"
            )
            return None

        # Check no duplicate epic + direction
        for pos in open_positions:
            if pos.epic == signal.epic and pos.direction == signal.direction:
                logger.info(
                    f"Already have {signal.direction} on {signal.epic} - skipping"
                )
                return None

        # Validate stop-loss exists
        if signal.stop_loss == 0 or signal.stop_loss == signal.entry_price:
            logger.warning(f"Invalid stop-loss for {signal.epic} - rejecting")
            return None

        # Validate R:R ratio
        if signal.risk_reward_ratio < RISK.MIN_RISK_REWARD:
            logger.info(
                f"R:R too low ({signal.risk_reward_ratio:.2f}) "
                f"for {signal.epic} - rejecting"
            )
            return None

        # Check market is open
        if not market_info.market_open:
            logger.info(f"Market closed for {signal.epic} - rejecting")
            return None

        # Calculate position size
        size = PositionSizer.calculate_size(
            equity=equity,
            risk_pct=RISK.MAX_RISK_PER_TRADE,
            entry_price=signal.entry_price,
            stop_loss_price=signal.stop_loss,
            market_info=market_info,
        )

        if size <= 0:
            logger.warning(f"Position size is 0 for {signal.epic} - rejecting")
            return None

        risk_amount = equity * RISK.MAX_RISK_PER_TRADE
        self._trades_today += 1

        logger.info(
            f"APPROVED: {signal.direction} {signal.epic} | "
            f"size={size} | SL={signal.stop_loss:.5f} | "
            f"TP={signal.take_profit:.5f} | R:R={signal.risk_reward_ratio:.2f}"
        )

        return ApprovedTrade(
            signal=signal,
            size=size,
            risk_amount=risk_amount,
        )

    def kill(self):
        self.is_active = False
        logger.critical("KILL SWITCH ACTIVATED - no more trades will be executed")

    def resume(self):
        self.is_active = True
        logger.info("Kill switch deactivated - trading resumed")
