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
        quote_to_account_rate: float = 1.0,
    ) -> float:
        """Size a position so that hitting the SL loses `risk_pct` of equity.

        For Capital.com CFDs: P&L (in QUOTE currency) = size * price_change.
        When the instrument's quote currency differs from the account
        currency, the loss must be converted:
            loss_account = size * sl_distance * quote_to_account_rate
        so size = risk_amount / (sl_distance * quote_to_account_rate).

        `quote_to_account_rate` is the value of 1 unit of quote currency in
        account currency (e.g. account=USD, USDJPY quote=JPY -> rate =
        USD per JPY ~ 0.0067). The caller MUST pass the real rate for
        non-account-currency quotes; RiskManager rejects such signals when no
        rate is available instead of silently mis-sizing them.
        """
        if quote_to_account_rate <= 0:
            logger.warning("Invalid quote_to_account_rate, cannot size position")
            return 0.0

        risk_amount = equity * risk_pct
        sl_distance = abs(entry_price - stop_loss_price)

        if sl_distance == 0:
            logger.warning("SL distance is 0, cannot size position")
            return 0.0

        size = risk_amount / (sl_distance * quote_to_account_rate)

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
            f"SL_dist={sl_distance:.5f}, quote_rate={quote_to_account_rate}, "
            f"risk_amount={risk_amount:.2f})"
        )
        return size
