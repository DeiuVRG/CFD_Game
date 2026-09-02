"""Sentinel configuration (env overrides documented in sentinel/README.md)."""
import os
import sys
from dataclasses import dataclass, field
from typing import List, Optional

from dotenv import load_dotenv

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SENTINEL_DIR = os.path.join(REPO_ROOT, "sentinel")
EXECUTION_CAPITAL_DIR = os.path.join(REPO_ROOT, "execution_capital")

if REPO_ROOT not in sys.path:          # for `common.indicators`
    sys.path.insert(0, REPO_ROOT)

# Secrets: gold_monitor/.env is the single recommended place; the other two
# locations are honoured for convenience (first value found wins).
for _p in (os.path.join(REPO_ROOT, "gold_monitor", ".env"),
           os.path.join(EXECUTION_CAPITAL_DIR, ".env"),
           os.path.join(REPO_ROOT, ".env")):
    if os.path.exists(_p):
        load_dotenv(_p, override=False)


def _env_bool(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    if v is None or v == "":
        return default
    return v.strip().lower() in ("1", "true", "yes", "on")


@dataclass
class InstrumentMap:
    signal_name: str         # gold_monitor SYMBOL_DISPLAY = `instrument` in signals.db
    epic: str                # Capital.com epic (verify: python -m sentinel.main --markets gold)
    resolution: str = "HOUR"  # Capital.com candle resolution used for context


@dataclass
class SentinelConfig:
    signals_db: str = os.path.join(REPO_ROOT, "gold_monitor", "data", "signals.db")
    decisions_db: str = os.path.join(SENTINEL_DIR, "data", "decisions.db")
    instruments: List[InstrumentMap] = field(default_factory=lambda: [
        InstrumentMap("XAU/USD (Gold)", os.getenv("SENTINEL_EPIC_GOLD", "GOLD")),
        InstrumentMap("BTC/USD (Bitcoin)", os.getenv("SENTINEL_EPIC_BTC", "BTCUSD")),
    ])

    # Cadence
    poll_interval_sec: int = 60          # new-signal poll
    review_interval_sec: int = 900       # open-position review (LLM call per position)
    signal_max_age_sec: int = 900        # older signals are stale -> never executed
    research_ttl_sec: int = 3600         # market research reused within this window
    candles_for_context: int = 48

    # Claude (the user asked for Claude Fable 5.1 explicitly)
    model: str = os.getenv("SENTINEL_MODEL", "claude-fable-5-1")
    # Which brain talks to Claude:
    #   "agent_sdk" (default) - Claude Agent SDK over the locally logged-in
    #       Claude Code CLI -> uses the Claude Pro/Max SUBSCRIPTION allowance.
    #       Needs `claude` installed and logged in; ANTHROPIC_API_KEY unset.
    #   "api" - Anthropic API via the anthropic SDK -> needs ANTHROPIC_API_KEY
    #       and prepaid API credits (a subscription does not include the API).
    brain: str = os.getenv("SENTINEL_BRAIN", "agent_sdk").strip().lower()
    sdk_fallback_model: str = os.getenv("SENTINEL_FALLBACK_MODEL", "")
    sdk_research_max_turns: int = 8
    effort_research: str = "medium"
    effort_decision: str = "high"
    max_tokens: int = 8000
    web_search: bool = _env_bool("SENTINEL_WEB_SEARCH", True)
    web_search_max_uses: int = 4

    # Hard limits - enforced in rules.py; the model cannot exceed them
    max_risk_per_trade: float = 0.01     # of equity, at size_fraction = 1.0
    max_daily_loss: float = 0.03         # of day-start equity -> no new trades
    max_trades_per_day: int = 5
    max_concurrent_positions: int = 2
    min_risk_reward: float = 1.0

    # Modes
    dry_run: bool = _env_bool("SENTINEL_DRY_RUN", False)
    capital_mode: str = os.getenv("CAPITAL_MODE", "demo").strip().lower()
    discord_webhook: str = os.getenv("DISCORD_WEBHOOK_URL", "")

    def epic_for(self, signal_name: str) -> Optional[InstrumentMap]:
        for m in self.instruments:
            if m.signal_name == signal_name:
                return m
        return None
