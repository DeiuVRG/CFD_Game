"""Decision contracts between the model and the sentinel.

The model answers with JSON constrained by these schemas (structured
outputs); the sentinel validates with pydantic and then applies the hard
rules in rules.py. An unparsable answer is treated as "no decision".
"""
import json
import logging
from typing import List, Literal, Optional

from pydantic import BaseModel, Field, ValidationError

logger = logging.getLogger(__name__)


class OpenDecision(BaseModel):
    action: Literal["APPROVE", "VETO"]
    size_fraction: float = Field(ge=0.0, le=1.0)   # x the max risk-based size
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str
    risks: List[str] = Field(default_factory=list)


class ManageDecision(BaseModel):
    action: Literal["HOLD", "CLOSE", "TIGHTEN_SL"]
    new_stop_loss: Optional[float] = None
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str


OPEN_SCHEMA = {
    "type": "object",
    "properties": {
        "action": {"type": "string", "enum": ["APPROVE", "VETO"]},
        "size_fraction": {"type": "number", "minimum": 0, "maximum": 1},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "rationale": {"type": "string"},
        "risks": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["action", "size_fraction", "confidence", "rationale", "risks"],
    "additionalProperties": False,
}

MANAGE_SCHEMA = {
    "type": "object",
    "properties": {
        "action": {"type": "string", "enum": ["HOLD", "CLOSE", "TIGHTEN_SL"]},
        "new_stop_loss": {"type": ["number", "null"]},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "rationale": {"type": "string"},
    },
    "required": ["action", "new_stop_loss", "confidence", "rationale"],
    "additionalProperties": False,
}


def _parse(model_cls, text: str):
    try:
        return model_cls.model_validate(json.loads(text))
    except (json.JSONDecodeError, ValidationError, TypeError) as e:
        logger.warning(f"Unparsable {model_cls.__name__}: {e}; text={text[:200]!r}")
        return None


def parse_open(text: str) -> Optional[OpenDecision]:
    return _parse(OpenDecision, text)


def parse_manage(text: str) -> Optional[ManageDecision]:
    return _parse(ManageDecision, text)
