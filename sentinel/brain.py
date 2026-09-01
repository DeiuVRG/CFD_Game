"""The model side of the sentinel: market research (with web search) and
structured APPROVE/VETO and HOLD/CLOSE/TIGHTEN_SL decisions.

Claude Fable 5.1 is used at the user's explicit request. Safety
classifiers can decline a request (HTTP 200, stop_reason "refusal"); the
server-side `fallbacks: "default"` opt-in re-runs such a request on
Anthropic's recommended fallback model automatically. A refusal of the
whole chain, an API error or an unparsable answer all yield None - and the
agent fails CLOSED on None (no trade / no change).
"""
import logging
from dataclasses import dataclass
from typing import Optional, Protocol

from sentinel.config import SentinelConfig
from sentinel.schema import (MANAGE_SCHEMA, OPEN_SCHEMA, ManageDecision,
                             OpenDecision, parse_manage, parse_open)

logger = logging.getLogger(__name__)

FALLBACK_BETA = "server-side-fallback-2026-07-01"
WEB_SEARCH_TOOL = "web_search_20260209"
MAX_PAUSE_CONTINUATIONS = 3

SYSTEM_COMMON = """You are the sentinel of a small CFD trading system that runs on a Capital.com DEMO account. Its purpose is to collect evidence, not money: every decision you make is logged and later scored against the plain deterministic path, so decide as if the account were real, and explain yourself so a reviewer can judge you.

The deterministic pipeline (an XGBoost model on completed 1h candles, an ADX regime gate, ATR-based stop-loss/take-profit, a cost and risk:reward filter) produces the signals. You never invent trades: you may only APPROVE or VETO a signal, scale its size down, and later HOLD, CLOSE or TIGHTEN the stop of a position. Hard limits (risk per trade, daily loss, trades per day, concurrent positions) are enforced in code after you answer; do not try to work around them.

Judge with the evidence in front of you: the signal's own confidence and regime, the market snapshot, the pipeline's recent track record, the account state and the research brief. Prefer VETO when the evidence conflicts (e.g. a scheduled high-impact event minutes away, a weak track record on this instrument, a regime the model has not seen). Write the rationale in Romanian, in at most 500 characters, leading with the decisive fact."""

SYSTEM_OPEN = SYSTEM_COMMON + """

Answer ONLY with the JSON object described by the schema: action APPROVE or VETO; size_fraction between 0 and 1 (1 = the full risk-based size, use less when conviction is partial); confidence between 0 and 1; rationale; risks (short list)."""

SYSTEM_MANAGE = SYSTEM_COMMON + """

You are reviewing an OPEN demo position. Answer ONLY with the JSON object described by the schema: action HOLD, CLOSE or TIGHTEN_SL; new_stop_loss (a number only for TIGHTEN_SL - it must move the stop in the position's favour and stay on the correct side of the current price, otherwise it is ignored); confidence; rationale. The take-profit and the original stop are the validated exits: HOLD is the default, CLOSE and TIGHTEN_SL need a concrete reason (adverse news, regime break, target nearly reached)."""

SYSTEM_RESEARCH = """You research the market context for a CFD instrument for a trading sentinel. Use web search to find what matters for the next 24-48 hours: scheduled high-impact macro events (central banks, CPI, NFP, ...), breaking news moving this instrument, notable positioning or sentiment, and anything that would make an automated technical signal unreliable right now. Be factual, cite the source and time of each item, and finish with a 3-line summary: bullish factors / bearish factors / event risk in the next 24h. Under 400 words. Write in English."""


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    model: str = ""


@dataclass
class Research:
    brief: str
    usage: Usage
    refused: bool = False


class Brain(Protocol):
    def research(self, instrument: str, snapshot: str) -> Research: ...
    def decide_open(self, context: str, research: str) -> tuple: ...
    def decide_manage(self, context: str, research: str) -> tuple: ...


class NullBrain:
    """No model: every signal is approved at full size and positions are
    held. Use it to run the deterministic path alone (a control group) or
    when no API key is available."""
    def research(self, instrument: str, snapshot: str) -> Research:
        return Research(brief="", usage=Usage())

    def decide_open(self, context: str, research: str):
        return OpenDecision(action="APPROVE", size_fraction=1.0, confidence=0.5,
                            rationale="NullBrain: deterministic pass-through", risks=[]), Usage()

    def decide_manage(self, context: str, research: str):
        return ManageDecision(action="HOLD", new_stop_loss=None, confidence=0.5,
                              rationale="NullBrain: hold"), Usage()


def _text_of(response) -> str:
    return "\n".join(b.text for b in response.content if getattr(b, "type", "") == "text")


def _usage_of(response) -> Usage:
    u = getattr(response, "usage", None)
    return Usage(
        input_tokens=int(getattr(u, "input_tokens", 0) or 0),
        output_tokens=int(getattr(u, "output_tokens", 0) or 0),
        model=str(getattr(response, "model", "") or ""),
    )


class ClaudeBrain:
    def __init__(self, cfg: SentinelConfig, client=None):
        self.cfg = cfg
        if client is None:
            import anthropic
            client = anthropic.Anthropic()
        self.client = client

    def _create(self, **kwargs):
        return self.client.beta.messages.create(
            model=self.cfg.model,
            max_tokens=self.cfg.max_tokens,
            betas=[FALLBACK_BETA],
            fallbacks="default",
            **kwargs,
        )

    @staticmethod
    def _refused(response) -> bool:
        if getattr(response, "stop_reason", None) != "refusal":
            return False
        details = getattr(response, "stop_details", None)
        logger.warning(f"Model refusal (category={getattr(details, 'category', None)})")
        return True

    def research(self, instrument: str, snapshot: str) -> Research:
        tools = []
        if self.cfg.web_search:
            tools = [{"type": WEB_SEARCH_TOOL, "name": "web_search",
                      "max_uses": self.cfg.web_search_max_uses}]
        user = (f"Instrument: {instrument}\nCurrent technical snapshot (1h candles):\n"
                f"{snapshot}\n\nResearch the market context now.")
        messages = [{"role": "user", "content": user}]
        response = None
        for _ in range(MAX_PAUSE_CONTINUATIONS + 1):
            response = self._create(
                system=[{"type": "text", "text": SYSTEM_RESEARCH,
                         "cache_control": {"type": "ephemeral"}}],
                messages=messages, tools=tools,
                output_config={"effort": self.cfg.effort_research},
            )
            if getattr(response, "stop_reason", None) != "pause_turn":
                break
            messages = [messages[0], {"role": "assistant", "content": response.content}]
        if self._refused(response):
            return Research(brief="", usage=_usage_of(response), refused=True)
        return Research(brief=_text_of(response).strip(), usage=_usage_of(response))

    def _decide(self, system: str, schema: dict, context: str, research: str, parser):
        user = f"Context (JSON):\n{context}\n\nResearch brief:\n{research or '(none)'}"
        response = self._create(
            system=[{"type": "text", "text": system,
                     "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": user}],
            output_config={"effort": self.cfg.effort_decision,
                           "format": {"type": "json_schema", "schema": schema}},
        )
        usage = _usage_of(response)
        if self._refused(response):
            return None, usage
        return parser(_text_of(response)), usage

    def decide_open(self, context: str, research: str):
        return self._decide(SYSTEM_OPEN, OPEN_SCHEMA, context, research, parse_open)

    def decide_manage(self, context: str, research: str):
        return self._decide(SYSTEM_MANAGE, MANAGE_SCHEMA, context, research, parse_manage)
