"""The v3 exit rule for ONE candle of an open position - the single shared
implementation used by the backtester (historical simulation) and by the
live position tracker (hypothetical outcomes of emitted signals), so that
what is collected live is judged by exactly the rules that were validated.

Rules (see ai/backtester.py module docstring):
  - SL/TP are checked against the candle HIGH/LOW (wicks count);
  - a candle that touches both SL and TP is resolved as SL (conservative);
  - an open beyond SL is a gap: the exit fills at the OPEN (worse than SL);
  - TP fills exactly at the TP level, never better.
"""
from typing import Optional, Tuple

EXIT_GAP_SL = "GAP_SL"
EXIT_SL = "SL"
EXIT_TP = "TP"


def v3_exit(direction: str, stop_loss: float, take_profit: float,
            o: float, h: float, l: float) -> Tuple[Optional[float], str]:
    """Return (exit_price, reason) if the position exits on this candle,
    else (None, "")."""
    if direction == "BUY":
        if o <= stop_loss:
            return o, EXIT_GAP_SL
        if l <= stop_loss:
            return stop_loss, EXIT_SL
        if h >= take_profit:
            return take_profit, EXIT_TP
    else:  # SELL
        if o >= stop_loss:
            return o, EXIT_GAP_SL
        if h >= stop_loss:
            return stop_loss, EXIT_SL
        if l <= take_profit:
            return take_profit, EXIT_TP
    return None, ""
