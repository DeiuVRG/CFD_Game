import csv
import logging
import os
from datetime import datetime

from broker.models import TradeConfirmation
from engine.signal import ApprovedTrade

logger = logging.getLogger(__name__)


class TradeLogger:
    HEADERS = [
        "timestamp", "epic", "direction", "size", "entry_price",
        "stop_loss", "take_profit", "strategy", "strength",
        "deal_id", "status", "reason",
    ]

    def __init__(self, output_dir: str = "output"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.filepath = os.path.join(output_dir, "trades.csv")

        if not os.path.exists(self.filepath):
            with open(self.filepath, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(self.HEADERS)

    def log_trade(self, trade: ApprovedTrade, confirmation: TradeConfirmation):
        row = [
            datetime.utcnow().isoformat(),
            trade.signal.epic,
            trade.signal.direction,
            trade.size,
            confirmation.level or trade.signal.entry_price,
            trade.signal.stop_loss,
            trade.signal.take_profit,
            trade.signal.strategy_name,
            trade.signal.strength,
            confirmation.deal_id,
            confirmation.status,
            confirmation.reason,
        ]

        with open(self.filepath, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(row)

        logger.info(
            f"TRADE: {trade.signal.direction} {trade.signal.epic} "
            f"size={trade.size} @ {confirmation.level:.5f} "
            f"[{confirmation.status}] ({trade.signal.strategy_name})"
        )
