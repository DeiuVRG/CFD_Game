"""Builds the compact, factual context the model reasons over."""
import json
from typing import Optional

import pandas as pd

from common.indicators import Indicators


def indicator_snapshot(df: Optional[pd.DataFrame]) -> dict:
    """Last-value snapshot of the classic indicators on the given candles."""
    if df is None or df.empty or len(df) < 30:
        return {}
    c, h, l = df["close"], df["high"], df["low"]
    rsi = Indicators.rsi(c, 14).iloc[-1]
    adx = Indicators.adx(h, l, c, 14).iloc[-1]
    atr = Indicators.atr(h, l, c, 14).iloc[-1]
    ema9 = Indicators.ema(c, 9).iloc[-1]
    ema21 = Indicators.ema(c, 21).iloc[-1]
    _, _, hist = Indicators.macd(c)
    last = float(c.iloc[-1])
    n24 = min(24, len(c) - 1)

    def f(x, nd=4):
        return None if pd.isna(x) else round(float(x), nd)

    return {
        "last_close": round(last, 4),
        "change_24_candles_pct": round((last / float(c.iloc[-1 - n24]) - 1) * 100, 2),
        "range_high": round(float(h.tail(24).max()), 4),
        "range_low": round(float(l.tail(24).min()), 4),
        "rsi_14": f(rsi, 1),
        "adx_14": f(adx, 1),
        "atr_14": f(atr),
        "atr_pct": f(atr / last * 100, 2) if not pd.isna(atr) else None,
        "ema9_vs_ema21_pct": f((ema9 / ema21 - 1) * 100, 2),
        "macd_hist": f(hist.iloc[-1]),
    }


def render_open_context(signal: dict, epic: str, snapshot: dict, stats: dict,
                        account: dict, cfg_limits: dict) -> str:
    entry, sl, tp = signal["entry_price"], signal["stop_loss"], signal["take_profit"]
    rr = abs(tp - entry) / abs(entry - sl) if sl and entry and sl != entry else 0
    data = {
        "signal": {
            "instrument": signal["instrument"], "epic": epic,
            "direction": signal["direction"], "time_utc": signal["ts_utc"],
            "entry": entry, "stop_loss": sl, "take_profit": tp,
            "risk_reward": round(rr, 2),
            "model_confidence": signal.get("confidence"),
            "probabilities": {"buy": signal.get("prob_buy"), "sell": signal.get("prob_sell"),
                              "hold": signal.get("prob_hold")},
            "adx_1h": signal.get("adx"), "regime": signal.get("regime"),
            "strategy": signal.get("strategy"), "model_version": signal.get("model_version"),
            "tier": signal.get("tier"),
        },
        "market_snapshot_1h": snapshot,
        "deterministic_track_record": stats,
        "account": account,
        "hard_limits": cfg_limits,
    }
    return json.dumps(data, ensure_ascii=False, indent=1, default=str)


def render_manage_context(trade: dict, position: dict, snapshot: dict,
                          account: dict, now_utc: str) -> str:
    entry = position.get("open_level") or trade.get("entry_price")
    price = snapshot.get("last_close") or entry
    if position.get("direction") == "BUY":
        unrealized_pct = (price / entry - 1) * 100 if entry else 0
    else:
        unrealized_pct = (1 - price / entry) * 100 if entry else 0
    data = {
        "position": {
            "epic": position.get("epic"), "deal_id": position.get("deal_id"),
            "direction": position.get("direction"), "size": position.get("size"),
            "entry": entry, "stop_loss": position.get("stop_level"),
            "take_profit": position.get("profit_level"),
            "opened_utc": trade.get("ts_utc"), "now_utc": now_utc,
            "unrealized_pnl_account_ccy": position.get("profit_loss"),
            "unrealized_pct": round(unrealized_pct, 3),
            "original_rationale": trade.get("llm_rationale"),
        },
        "market_snapshot_1h": snapshot,
        "account": account,
    }
    return json.dumps(data, ensure_ascii=False, indent=1, default=str)
