"""Brain backed by the Claude Agent SDK (Claude Code as a library).

Why it exists: a Claude Pro/Max subscription does not include the Claude
API (that is billed separately, per token, with prepaid credits). The Agent
SDK drives the locally logged-in Claude Code CLI, so with this brain the
sentinel runs on the subscription's usage allowance instead of API credits.
Keep ANTHROPIC_API_KEY unset on this path: an API key takes precedence over
the subscription login and would be billed per token.

Same contracts as brain.ClaudeBrain (research / decide_open / decide_manage),
same prompts, same JSON schemas; web research uses Claude Code's built-in
WebSearch/WebFetch tools.
"""
import asyncio
import logging
import os
from typing import Optional

from sentinel.brain import (SYSTEM_MANAGE, SYSTEM_OPEN, SYSTEM_RESEARCH, Research,
                            Usage)
from sentinel.config import SENTINEL_DIR, SentinelConfig
from sentinel.schema import MANAGE_SCHEMA, OPEN_SCHEMA, parse_manage, parse_open

logger = logging.getLogger(__name__)

RESEARCH_TOOLS = ["WebSearch", "WebFetch"]


class AgentSdkBrain:
    def __init__(self, cfg: SentinelConfig, query_fn=None, cwd: str = None):
        self.cfg = cfg
        if query_fn is None:
            from claude_agent_sdk import query as query_fn
        self._query = query_fn
        # An empty working directory: no CLAUDE.md, skills or memory leak in.
        self.cwd = cwd or os.path.join(SENTINEL_DIR, "data", "agent_cwd")
        os.makedirs(self.cwd, exist_ok=True)
        if os.environ.pop("ANTHROPIC_API_KEY", None):
            logger.warning("ANTHROPIC_API_KEY removed from this process: the Agent SDK "
                           "brain must use the Claude subscription login, not API billing")

    def _options(self, system: str, tools: list, max_turns: int, effort: str,
                 output_format: Optional[dict] = None):
        from claude_agent_sdk import ClaudeAgentOptions
        return ClaudeAgentOptions(
            model=self.cfg.model,
            fallback_model=self.cfg.sdk_fallback_model or None,
            system_prompt=system,
            tools=list(tools), allowed_tools=list(tools),
            max_turns=max_turns, effort=effort,
            setting_sources=[], cwd=self.cwd,
            output_format=output_format,
        )

    def _run(self, prompt: str, options):
        async def go():
            result = None
            async for m in self._query(prompt=prompt, options=options):
                if hasattr(m, "is_error") and hasattr(m, "structured_output"):
                    result = m
            return result
        try:
            return asyncio.run(go())
        except Exception as e:      # CLI missing, login expired, transport error...
            logger.error(f"Agent SDK query failed: {e}")
            return None

    @staticmethod
    def _usage(result) -> Usage:
        u = getattr(result, "usage", None) or {}
        return Usage(input_tokens=int(u.get("input_tokens", 0) or 0),
                     output_tokens=int(u.get("output_tokens", 0) or 0),
                     model="")

    @staticmethod
    def _failed(result) -> bool:
        if result is None or getattr(result, "is_error", False):
            return True
        if getattr(result, "stop_reason", None) == "refusal":
            logger.warning("Model refusal (Agent SDK)")
            return True
        return False

    def research(self, instrument: str, snapshot: str) -> Research:
        prompt = (f"Instrument: {instrument}\nCurrent technical snapshot (1h candles):\n"
                  f"{snapshot}\n\nResearch the market context now.")
        result = self._run(prompt, self._options(
            SYSTEM_RESEARCH, RESEARCH_TOOLS, self.cfg.sdk_research_max_turns,
            self.cfg.effort_research))
        if self._failed(result):
            return Research(brief="", usage=self._usage(result), refused=result is not None)
        usage = self._usage(result)
        usage.model = self.cfg.model
        return Research(brief=(result.result or "").strip(), usage=usage)

    def _decide(self, system: str, schema: dict, context: str, research: str, parser):
        prompt = f"Context (JSON):\n{context}\n\nResearch brief:\n{research or '(none)'}"
        result = self._run(prompt, self._options(
            system, [], 3, self.cfg.effort_decision,
            output_format={"type": "json_schema", "schema": schema}))
        usage = self._usage(result)
        usage.model = self.cfg.model
        if self._failed(result):
            return None, usage
        out = getattr(result, "structured_output", None)
        if isinstance(out, dict):
            import json
            return parser(json.dumps(out)), usage
        return parser(result.result or ""), usage

    def decide_open(self, context: str, research: str):
        return self._decide(SYSTEM_OPEN, OPEN_SCHEMA, context, research, parse_open)

    def decide_manage(self, context: str, research: str):
        return self._decide(SYSTEM_MANAGE, MANAGE_SCHEMA, context, research, parse_manage)
