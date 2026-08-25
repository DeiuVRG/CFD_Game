"""Game configuration: data segments, capital, costs, cadence, elimination
and recalibration rules. Mirrors the spec in docs/trading_game_prompt.md."""
from dataclasses import dataclass, field
from typing import List


# Default universe: liquid S&P 500 large caps with full 2015-2023 history.
DEFAULT_UNIVERSE = [
    "AAPL", "MSFT", "AMZN", "GOOGL", "JNJ", "JPM", "XOM", "PG", "KO", "PFE",
    "WMT", "DIS", "CSCO", "INTC", "VZ", "HD", "MCD", "BA", "CAT", "NKE",
]

MARKET_SYMBOL = "SPY"    # S&P 500 proxy (benchmark + regime trend)
VIX_SYMBOL = "^VIX"      # Volatility regime


@dataclass
class GameConfig:
    # --- Data segments (60/20/20) ---
    train_start: str = "2015-01-01"
    train_end: str = "2019-12-31"
    validation_start: str = "2020-01-01"
    validation_end: str = "2021-12-31"     # 24 months of competition
    test_start: str = "2022-01-01"
    test_end: str = "2023-12-31"           # unknown to players until the end

    universe: List[str] = field(default_factory=lambda: list(DEFAULT_UNIVERSE))

    # --- Capital & portfolio rules ---
    initial_capital: float = 100_000.0
    max_positions: int = 20
    min_trades_per_month: int = 1
    max_trades_per_month: int = 50

    # --- Realistic transaction costs ---
    commission_percent: float = 0.001      # 0.1% per transaction
    commission_fixed: float = 1.0          # minimum $1 per transaction
    spread_percent: float = 0.0005         # 0.05% bid-ask
    slippage_base: float = 0.0003          # 0.03% base slippage
    max_percent_daily_volume: float = 0.01  # liquidity: max 1% of ADV

    # --- Recalibration ---
    recalibration_every_months: int = 3
    max_weight_shift: float = 0.30         # max 30% change per indicator

    # --- Elimination ---
    elimination_gap: float = 0.20          # last >=20% behind second-last
    catastrophic_drawdown_equity: float = 0.50  # equity < 50% of initial

    # --- Scoring ---
    risk_free_rate: float = 0.02
    signal_buy_threshold: float = 0.3
    signal_sell_threshold: float = -0.3

    # --- Weight rules ---
    min_indicators: int = 3
    max_indicators: int = 20

    # --- Validation (test set) ---
    bootstrap_iterations: int = 1000
    permutation_iterations: int = 1000

    # --- Runtime ---
    seed: int = 42
    synthetic: bool = False                # offline fallback data
    data_cache_dir: str = "trading_game/data_cache"
