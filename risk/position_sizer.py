import logging

from broker.models import MarketInfo

logger = logging.getLogger(__name__)


class PositionSizer:

    @staticmethod
    def calculate_size(
        equity: float,
        risk_pct: float,
        entry_price: float,
        stop_loss_price: float,
        market_info: MarketInfo,
    ) -> float:
        risk_amount = equity * risk_pct
        sl_distance = abs(entry_price - stop_loss_price)

        if sl_distance == 0:
            logger.warning("SL distance is 0, cannot size position")
            return 0.0

        # For Capital.com CFDs: P&L = size * price_change
        # So size = risk_amount / SL_distance
        size = risk_amount / sl_distance

        # Clamp to broker max
        size = min(size, market_info.max_deal_size)

        # If size is below broker minimum, skip the trade
        if size < market_info.min_deal_size:
            logger.warning(
                f"Calculated size {size:.4f} below min {market_info.min_deal_size}"
            )
            return 0.0

        # Round to 2 decimal places
        size = round(size, 2)

        logger.debug(
            f"Position size: {size} (equity={equity}, risk={risk_pct*100}%, "
            f"SL_dist={sl_distance:.5f}, risk_amount={risk_amount:.2f})"
        )
        return size
