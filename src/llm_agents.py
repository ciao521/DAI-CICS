"""
llm_agents.py – LLM-powered agent prompts and AWS Bedrock API client.

Implements three agents for the discharge-coordination conference:
  - CareManagerAgent    (高疲労・高使命感のケアマネジャー)
  - DoctorAgent         (低疲労・高ベッド回転圧力の主治医)
  - PlannerAIAgent      (システム崩壊を防ぐ制度設計ファシリテーターAI)

Uses AWS Bedrock Runtime with Bearer Token authentication.
Model: cross-region inference profile for Claude claude-opus-4-5 (us-east-1).
"""
from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import requests

# ── Load .env from project root (safe: no-op if file not found) ─────────────
try:
    from dotenv import load_dotenv as _load_dotenv
    _load_dotenv(Path(__file__).parent.parent / ".env", override=False)
except ImportError:
    pass  # python-dotenv not installed; env vars must be set manually

# ──────────────────────────────────────────────────────────────
# Bedrock client
# ──────────────────────────────────────────────────────────────

BEDROCK_REGION = os.getenv("AWS_BEDROCK_REGION", "us-east-1")
# Claude Opus 4.6 cross-region inference profile ID on AWS Bedrock (us-east-1)
_DEFAULT_MODEL_ID = "us.anthropic.claude-opus-4-6-v1"

MAX_TOKENS = int(os.getenv("BEDROCK_MAX_TOKENS", "1024"))
TEMPERATURE = float(os.getenv("BEDROCK_TEMPERATURE", "0.7"))


def _get_endpoint() -> str:
    """Compute endpoint at call-time so BEDROCK_MODEL_ID env override works."""
    model_id = os.getenv("BEDROCK_MODEL_ID", _DEFAULT_MODEL_ID)
    region = os.getenv("AWS_BEDROCK_REGION", BEDROCK_REGION)
    return (
        f"https://bedrock-runtime.{region}.amazonaws.com"
        f"/model/{model_id}/invoke"
    )


def _get_bearer_token() -> str:
    """Read the bearer token from environment variable."""
    token = os.getenv("AWS_BEARER_TOKEN_BEDROCK", "")
    if not token:
        raise EnvironmentError(
            "AWS_BEARER_TOKEN_BEDROCK environment variable is not set. "
            "Export it before running: export AWS_BEARER_TOKEN_BEDROCK=bedrock-api-key-..."
        )
    return token


def call_bedrock(
    system_prompt: str,
    user_message: str,
    max_tokens: int = MAX_TOKENS,
    temperature: float = TEMPERATURE,
    retry: int = 3,
) -> str:
    """
    Call AWS Bedrock Runtime (Claude) with Bearer Token authentication.

    Returns the raw text content of the model's response.
    """
    token = _get_bearer_token()
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }

    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": max_tokens,
        "temperature": temperature,
        "system": system_prompt,
        "messages": [
            {"role": "user", "content": user_message},
        ],
    }

    endpoint = _get_endpoint()
    for attempt in range(retry):
        try:
            resp = requests.post(
                endpoint,
                headers=headers,
                data=json.dumps(body),
                timeout=60,
            )
            resp.raise_for_status()
            data = resp.json()
            # Claude Bedrock response structure
            return data["content"][0]["text"]
        except requests.exceptions.HTTPError as e:
            if resp.status_code in (429, 503) and attempt < retry - 1:
                time.sleep(2 ** attempt)
                continue
            raise RuntimeError(
                f"Bedrock API error ({resp.status_code}): {resp.text}"
            ) from e
        except Exception as e:
            if attempt < retry - 1:
                time.sleep(2 ** attempt)
                continue
            raise

    raise RuntimeError("Bedrock call failed after retries")


def _extract_json(raw: str) -> dict:
    """Extract the first JSON object from a response string.

    Handles:
    - Markdown code fences (```json ... ```)
    - Raw newlines / tab characters inside JSON string values
    - Trailing commas
    """
    # Strip markdown code fences
    text = re.sub(r"```(?:json)?", "", raw).strip()

    # Find the outermost { ... } block
    start = text.find("{")
    end = text.rfind("}") + 1
    if start == -1 or end == 0:
        raise ValueError(f"No JSON found in response: {raw[:300]}")
    json_str = text[start:end]

    # First try: direct parse
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        pass

    # Second try: replace raw control characters inside string literals
    # Replace literal newlines with \n escape, literal tabs with \t
    sanitized = re.sub(
        r'(?<=[":,\[{])\s*\n\s*(?=[^"{}:\[\]])',
        " ",
        json_str,
    )
    # More aggressive: escape raw newlines/tabs that appear inside " " boundaries
    # Walk the string and escape control chars inside strings
    in_string = False
    escape_next = False
    chars = []
    for ch in json_str:
        if escape_next:
            chars.append(ch)
            escape_next = False
            continue
        if ch == "\\":
            escape_next = True
            chars.append(ch)
            continue
        if ch == '"':
            in_string = not in_string
            chars.append(ch)
            continue
        if in_string and ch in ("\n", "\r", "\t"):
            chars.append("\\n" if ch == "\n" else ("\\r" if ch == "\r" else "\\t"))
            continue
        chars.append(ch)
    cleaned = "".join(chars)

    return json.loads(cleaned)


# ──────────────────────────────────────────────────────────────
# Agent data classes
# ──────────────────────────────────────────────────────────────

@dataclass
class AgentState:
    """Runtime state for an LLM agent."""
    name: str
    fatigue: float = 0.0
    altruism: float = 0.5
    extra_vars: dict = field(default_factory=dict)


# ──────────────────────────────────────────────────────────────
# Prompts (imported from prompts/ for easy editing)
# ──────────────────────────────────────────────────────────────
from prompts.care_manager import CARE_MANAGER_SYSTEM_PROMPT  # noqa: E402
from prompts.doctor import DOCTOR_SYSTEM_PROMPT              # noqa: E402
from prompts.planner_ai import PLANNER_SYSTEM_PROMPT         # noqa: E402

# ──────────────────────────────────────────────────────────────
# Care Manager Agent
# ──────────────────────────────────────────────────────────────


class CareManagerAgent:
    """LLM-backed care manager agent for discharge coordination conferences."""

    def __init__(self, state: AgentState | None = None) -> None:
        self.state = state or AgentState(
            name="CareManager",
            fatigue=0.88,
            altruism=0.80,
        )

    def respond(self, prior_utterance: str) -> dict:
        system = CARE_MANAGER_SYSTEM_PROMPT.format(
            fatigue=self.state.fatigue,
            altruism=self.state.altruism,
        )
        raw = call_bedrock(system, prior_utterance)
        result = _extract_json(raw)
        result["agent"] = "CareManager"
        result["fatigue"] = self.state.fatigue
        result["altruism"] = self.state.altruism
        return result


# ──────────────────────────────────────────────────────────────
# Doctor Agent
# ──────────────────────────────────────────────────────────────

class DoctorAgent:
    def __init__(self, state: AgentState | None = None) -> None:
        self.state = state or AgentState(
            name="Doctor",
            fatigue=0.40,
            altruism=0.90,
            extra_vars={
                "bed_pressure": 0.95,
                "idealism": 0.90,
            },
        )

    def respond(self, prior_utterance: str) -> dict:
        system = DOCTOR_SYSTEM_PROMPT.format(
            fatigue=self.state.fatigue,
            bed_pressure=self.state.extra_vars.get("bed_pressure", 0.95),
            idealism=self.state.extra_vars.get("idealism", 0.90),
        )
        raw = call_bedrock(system, prior_utterance)
        result = _extract_json(raw)
        result["agent"] = "Doctor"
        result["fatigue"] = self.state.fatigue
        return result


# ──────────────────────────────────────────────────────────────
# Planner AI Agent
# ──────────────────────────────────────────────────────────────

class PlannerAIAgent:
    def __init__(self, state: AgentState | None = None) -> None:
        self.state = state or AgentState(
            name="PlannerAI",
            fatigue=0.10,
            altruism=1.0,
            extra_vars={
                "cm_fatigue": 0.88,
                "gini_fatigue": 0.45,
                "discussion_status": "デッドロック寸前（医師は理想を要求し、ケアマネは過労で拒絶している状態：FC-C1の発現）",
            },
        )

    def respond(self, prior_utterance: str) -> dict:
        ev = self.state.extra_vars
        system = PLANNER_SYSTEM_PROMPT.format(
            cm_fatigue=ev.get("cm_fatigue", 0.88),
            gini_fatigue=ev.get("gini_fatigue", 0.45),
            discussion_status=ev.get("discussion_status", "デッドロック寸前"),
        )
        raw = call_bedrock(system, prior_utterance)
        result = _extract_json(raw)
        result["agent"] = "PlannerAI"
        result["applied_nudge"] = result.get("applied_nudge", "LOAD_BALANCING_AND_INCENTIVE")
        return result

    def update_metrics(self, cm_fatigue: float, gini_fatigue: float, status: str) -> None:
        self.state.extra_vars["cm_fatigue"] = cm_fatigue
        self.state.extra_vars["gini_fatigue"] = gini_fatigue
        self.state.extra_vars["discussion_status"] = status
